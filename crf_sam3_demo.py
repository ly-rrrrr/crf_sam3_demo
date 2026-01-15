"""
CRF-SAM3: Concept Recall & Spatio-Temporal Filtering with SAM 3

A new paradigm for Referring Video Object Segmentation (RVOS):
- Stage 1: Concept-Level Recall - SAM3 captures ALL candidate objects by category
- Stage 2: Visual Tube Construction - Build spatio-temporal representations with SoM
- Stage 3: Semantic Discrimination - MLLM scores and filters candidates

Key advantages:
1. High fault tolerance - recall all candidates first, filter later
2. Action understanding - full temporal context in Visual Tubes
3. Simple implementation - reuse pretrained SAM3 and MLLM
"""

import os
import json
import glob
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple
import torch
import cv2


def contains_color_attribute(expression: str) -> bool:
    """
    Detect if the referring expression contains color attributes.

    When color attributes are present, we MUST use bbox visualization instead of mask,
    because mask overlay will hide the original color information that MLLM needs to judge.

    Args:
        expression: Referring expression string

    Returns:
        True if expression contains color words, False otherwise
    """
    # Common color words (English)
    color_words = {
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'silver', 'golden', 'dark', 'light',
        'bright', 'pale', 'deep', 'navy', 'maroon', 'olive', 'cyan', 'magenta',
        'beige', 'tan', 'cream', 'ivory', 'colored', 'colour', 'color'
    }

    # Convert to lowercase and split into words
    words = expression.lower().split()

    # Check if any color word appears
    for word in words:
        # Remove punctuation for matching
        clean_word = word.strip('.,!?;:()[]{}"\'-')
        if clean_word in color_words:
            return True

    return False


class ConceptExtractor:
    """Stage 1a: Extract core category from complex referring expression"""

    def __init__(self, mllm_path: str, model_type: str = "qwen2.5",
                 use_qframe: bool = False, qframe_config: Dict = None,
                 category_prompt_path: str = "configs/category_num_targets_inference.txt"):
        """
        Args:
            mllm_path: Path to MLLM (Qwen2.5-VL, Qwen3-VL, or Qwen3-VL-Thinking)
            model_type: Type of model ("qwen2.5", "qwen3", or "qwen3-thinking")
            use_qframe: Whether to use Q-Frame for keyframe selection
            qframe_config: Q-Frame configuration dict with 'clip_model_path', 'num_frames', 'tau'
            category_prompt_path: Path to category/num_targets inference prompt template
        """
        from transformers import AutoProcessor

        self.model_type = model_type.lower()
        self.use_qframe = use_qframe
        self.qframe_config = qframe_config if qframe_config else {}

        # Load category inference prompt template
        self.category_prompt_template = self._load_prompt_template(category_prompt_path)
        print(f"[ConceptExtractor] Loaded category prompt from: {category_prompt_path}")

        print(f"[ConceptExtractor] Loading {self.model_type.upper()} from {mllm_path}")

        # Load model based on type
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            from transformers import Qwen3VLForConditionalGeneration
            self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
                mllm_path,
                dtype="auto",
                device_map="auto",
            ).eval()
            self.process_vision_info = None
        elif self.model_type == "qwen2.5":
            from transformers import Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mllm_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            self.process_vision_info = process_vision_info
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'qwen2.5', 'qwen3', or 'qwen3-thinking'")

        self.processor = AutoProcessor.from_pretrained(
            mllm_path,
            trust_remote_code=(self.model_type == "qwen2.5")
        )

        # Initialize Q-Frame selector if needed
        self.qframe_selector = None
        if self.use_qframe:
            from mapp_sam3_demo import QFrameSelector
            clip_model_path = self.qframe_config.get('clip_model_path')
            device = self.qframe_config.get('device', 'cuda')

            if not clip_model_path:
                raise ValueError("Q-Frame requires 'clip_model_path' in qframe_config")

            # Convert 'auto' to 'cuda' for Long-CLIP
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            print(f"[ConceptExtractor] Initializing Q-Frame selector...")
            self.qframe_selector = QFrameSelector(clip_model_path, device, apply_rebalancing=False)

        print(f"[ConceptExtractor] MLLM loaded successfully ({self.model_type.upper()})")
        if self.use_qframe:
            print(f"[ConceptExtractor] Q-Frame keyframe selection: enabled")

    def _load_prompt_template(self, prompt_path: str) -> str:
        """Load prompt template from file."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_category(self, referring_expression: str, video_path: str = None) -> Tuple[str, int, Dict]:
        """
        Extract base category and estimated target count from expression + multiple frames.

        Args:
            referring_expression: Complex expression like "the dog that stops and sits"
            video_path: Path to video frames directory (for visual context)

        Returns:
            (category, num_targets, reasoning): e.g., ("dog", 1, {...})
            reasoning contains sub_conditions and final_rule_applied
        """
        # Load frames for visual context
        sampled_frame_paths = []
        if video_path:
            frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
            if len(frame_files) > 0:
                if self.use_qframe and self.qframe_selector:
                    # Use Q-Frame to select semantically relevant frames
                    print(f"[ConceptExtractor] Using Q-Frame to select keyframes...")

                    # Load all video frames
                    video_frames = load_video_frames(video_path)

                    # Get number of frames from config (default to 8)
                    num_frames = self.qframe_config.get('num_frames', 8)
                    tau = self.qframe_config.get('tau', 0.8)

                    # Select keyframes using Q-Frame
                    qframe_indices = self.qframe_selector.select_keyframes(
                        video_frames,
                        referring_expression,
                        num_candidates=num_frames,
                        tau=tau
                    )

                    # Get frame paths
                    sampled_frame_paths = [frame_files[i] for i in qframe_indices]

                    frame_names = [os.path.basename(p) for p in sampled_frame_paths]
                    print(f"[ConceptExtractor] Q-Frame selected {len(sampled_frame_paths)} frames: {frame_names}")

                else:
                    # Original: Sample 3 frames: first, middle, last (COMMENTED OUT)
                    # This provides temporal coverage to avoid single-frame bias
                    # num_frames = len(frame_files)
                    # indices = [0, num_frames // 2, num_frames - 1]
                    # sampled_frame_paths = [frame_files[i] for i in indices]

                    # Use uniform sampling as fallback (8 frames)
                    num_frames_to_sample = 8
                    num_frames = len(frame_files)
                    if num_frames <= num_frames_to_sample:
                        sampled_frame_paths = frame_files
                    else:
                        step = num_frames / num_frames_to_sample
                        indices = [int(i * step) for i in range(num_frames_to_sample)]
                        sampled_frame_paths = [frame_files[i] for i in indices]

                    frame_names = [os.path.basename(p) for p in sampled_frame_paths]
                    print(f"[ConceptExtractor] Using {len(sampled_frame_paths)} uniformly sampled frames: {frame_names}")


        if sampled_frame_paths:
            # Use configured prompt template
            prompt = self.category_prompt_template.format(referring_expression=referring_expression)
#             prompt = f"""Look at these images sampled from a video (first, middle, and last frames) and the referring expression below.
#
# Referring Expression: "{referring_expression}"
#
# Task:
# 1. **Identify the grammatical subject** (the main noun category being referred to)
# 2. **Analyze language cues** about quantity (singular/plural, numbers, determiners)
# 3. **Verify visually** how many objects matching the description actually exist in the frames
# 4. **Balance language and vision** to determine the final count
#
# CRITICAL PRINCIPLE: Use BOTH language structure AND visual evidence together!
# - Language tells you WHAT to look for and HOW to count
# - Vision tells you HOW MANY actually exist matching that description
# - The final count should respect BOTH constraints
#
# STEP-BY-STEP ANALYSIS:
#
# Step 1: Parse the language structure
#    Identify these components:
#    a) Main noun (singular or plural?)
#       - "dog" / "rabbit" → singular form
#       - "dogs" / "rabbits" / "Monkeys" → plural form
#
#    b) Determiners and quantifiers
#       - "the" / "a" / "an" → could be singular, but check noun form!
#       - Explicit numbers ("4 lizards", "two bears") → use that exact number
#       - No determiner ("Monkeys on the left") → check plural noun + visual count
#
#    c) Spatial/attribute modifiers
#       - Specific location ("on the left", "in the distance") → limits scope
#       - Attributes ("white rabbit", "black rabbit") → filters by attribute
#       - Actions ("sitting down", "jumping") → filters by behavior
#
# Step 2: Visual verification
#    In the provided frames, count how many objects:
#    - Match the category (main noun)
#    - Match the spatial constraint (if any)
#    - Match the attribute/behavior (if any)
#
#    This gives you the VISUAL CANDIDATE COUNT.
#
# Step 3: Reconcile language and vision
#    Apply these rules in order:
#
#    Rule 1: **Explicit number trumps all**
#       "4 lizards" → num_targets = 4 (even if you see more or fewer)
#
#    Rule 2: **Plural noun + spatial/attribute modifier**
#       "Monkeys on the left" = plural + location
#       → Count ALL visually matching objects in that location
#       Example: If you see 2 monkeys on the left → num_targets = 2
#
#    Rule 3: **Singular noun with singular determiner**
#       "the elephant" / "a rabbit" = singular noun + singular determiner
#       → num_targets = 1
#
#    Rule 4: **Plural noun without specific constraint**
#       "elephants moving" = plural, no specific location/attribute
#       → Count visually matching objects (minimum 2, use what you see)
#
#    Rule 5: **Attribute modifier alone does NOT force singular**
#       "white rabbit" = attribute + singular noun → likely 1
#       "black rabbit eating" = attribute + singular noun → likely 1
#       BUT: "white rabbits" = attribute + plural noun → count visually
#
# COMMON MISTAKES TO AVOID:
# ❌ "Monkeys on the left" → num_targets = 1 (wrong! plural noun should check visual count in that region)
# ✓ "Monkeys on the left" → If 2 monkeys visible on left → num_targets = 2
#
# ❌ "white rabbit jumping" → num_targets = 2 (wrong! singular noun despite seeing multiple rabbits)
# ✓ "white rabbit jumping" → num_targets = 1 (singular noun form)
#
# ❌ "elephants moving forward" → num_targets = 2 (wrong! should check visual count)
# ✓ "elephants moving forward" → If 3 elephants visible → num_targets = 3
#
# Output JSON:
# {{
#   "category": "<base noun in singular form>",
#   "num_targets": <final count after reconciling language + vision>,
#   "reasoning": "<explain: 1) language structure (noun form, determiners), 2) visual count in frames, 3) how you reconciled them>"
# }}
#
# Examples:
# - "the elephant that turns around" → {{"category": "elephant", "num_targets": 1, "reasoning": "Language: singular noun 'elephant' + determiner 'the'. Visual: multiple elephants visible. Reconciliation: singular form forces count=1."}}
# - "Monkeys on the left sitting down without moving" → {{"category": "monkey", "num_targets": 2, "reasoning": "Language: plural noun 'Monkeys' + location 'on the left'. Visual: 2 monkeys visible on left side sitting. Reconciliation: plural + location → count visually matching objects = 2."}}
# - "4 lizards moving around" → {{"category": "lizard", "num_targets": 4, "reasoning": "Language: explicit number '4'. Visual: lizards visible. Reconciliation: explicit number trumps all → count=4."}}
# - "white rabbit jumping" → {{"category": "rabbit", "num_targets": 1, "reasoning": "Language: singular noun 'rabbit' + attribute 'white'. Visual: may see multiple rabbits. Reconciliation: singular noun form → count=1."}}
# - "elephants walking ahead" → {{"category": "elephant", "num_targets": 3, "reasoning": "Language: plural noun 'elephants' + direction 'ahead'. Visual: 3 elephants visible walking forward. Reconciliation: plural without explicit number → count visually matching = 3."}}
# """

            # Build multi-image content
            content = []
            for frame_path in sampled_frame_paths:
                content.append({"type": "image", "image": frame_path})
            content.append({"type": "text", "text": prompt})

            messages = [{
                "role": "user",
                "content": content
            }]
        else:
            # Fallback to text-only if no video path provided
            print(f"[ConceptExtractor] Warning: No video path provided, using text-only extraction")
            prompt = f"""Extract the core object category from this referring expression.

Referring Expression: "{referring_expression}"

Task:
1. Identify the base noun category (e.g., "dog", "cow", "person")
2. Determine if singular (1) or plural (2+) based on language cues

Output JSON:
{{
  "category": "<base noun>",
  "num_targets": <1 or estimated number>,
  "reasoning": "<brief explanation>"
}}

Examples:
- "the dog that stops running" → {{"category": "dog", "num_targets": 1}}
- "elephants moving forward" → {{"category": "elephant", "num_targets": 2}}
- "the white cow" → {{"category": "cow", "num_targets": 1}}
"""
            messages = [{"role": "user", "content": prompt}]

        # Process inputs based on model type
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            # Qwen3-VL: Simplified API with do_sample_frames=False for image lists
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False  # Required for list of images
            )
            inputs = inputs.to(self.mllm.device)
        else:
            # Qwen2.5-VL: Requires process_vision_info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info if images are provided
            if sampled_frame_paths:
                image_inputs, video_inputs, video_kwargs = self.process_vision_info(messages, return_video_kwargs=True)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(self.mllm.device)
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt"
                ).to(self.mllm.device)

        with torch.no_grad():
            if self.model_type == "qwen3-thinking":
                # Qwen3-VL-Thinking specific generation parameters
                # Uses thinking mechanism with higher diversity sampling
                output_ids = self.mllm.generate(
                    **inputs,
                    max_new_tokens=40960,  # Large output for thinking process
                    do_sample=True,
                    top_p=0.95,
                    top_k=20,
                    temperature=1.0,  # Higher temperature for diverse thinking
                    repetition_penalty=1.0,
                )
            elif self.model_type == "qwen3":
                # Qwen3-VL specific generation parameters (lower temperature for concept extraction)
                output_ids = self.mllm.generate(
                    **inputs,
                    max_new_tokens=256,  # Increased to avoid truncation
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    temperature=0.3,  # Lower temperature for more deterministic output
                    repetition_penalty=1.0,
                )
            else:
                # Qwen2.5-VL generation parameters
                output_ids = self.mllm.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1
                )

        # Decode only the generated part (trim input)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"\n[ConceptExtractor] Raw response:\n{response}\n")

        # Parse JSON
        import re
        import json

        # Try to fix incomplete JSON (add missing closing braces)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            # Try to fix incomplete JSON by adding closing brace
            if response.strip().startswith('{') and not response.strip().endswith('}'):
                response = response.strip() + '"}'
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_match.group())
            result = json.loads(json_str)
            category = result["category"]
            num_targets = result.get("num_targets", 1)
            reasoning = result.get("reasoning", {})

            # Extract sub_conditions and final_rule_applied
            sub_conditions = reasoning.get("sub_conditions", []) if isinstance(reasoning, dict) else []
            final_rule_applied = reasoning.get("final_rule_applied", "N/A") if isinstance(reasoning, dict) else "N/A"

            print(f"[ConceptExtractor] Extracted category: '{category}'")
            print(f"[ConceptExtractor] Estimated targets: {num_targets}")
            print(f"[ConceptExtractor] Sub-conditions: {sub_conditions}")
            print(f"[ConceptExtractor] Final rule: {final_rule_applied}")

            # CRITICAL FALLBACK: If num_targets is 0, default to 1
            # Reasoning: If the expression mentions an object, there should be at least 1 target
            # num_targets=0 usually means MLLM couldn't visually confirm the object,
            # but we should still try to detect it (e.g., "black rabbit" in poor lighting)
            if num_targets == 0:
                print(f"[ConceptExtractor] ⚠ WARNING: num_targets=0 detected!")
                print(f"[ConceptExtractor] This usually means MLLM couldn't visually confirm the object.")
                print(f"[ConceptExtractor] Applying fallback: num_targets=0 → num_targets=1")
                print(f"[ConceptExtractor] Rationale: The expression '{referring_expression}' refers to at least 1 object.")
                num_targets = 1

            return category, num_targets, {
                "sub_conditions": sub_conditions,
                "final_rule_applied": final_rule_applied
            }
        else:
            raise ValueError(f"Failed to parse MLLM response: {response}")


class SAM3ConceptRecall:
    """Stage 1b: Use SAM3 to recall all instances of a category"""

    def __init__(self, sam3_checkpoint: str):
        """
        Args:
            sam3_checkpoint: Path to SAM3 checkpoint
        """
        import sys
        sys.path.append('/mnt/csip-200/share/kongjiawei/sam3-main')
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        print(f"[SAM3Recall] Loading SAM3 from {sam3_checkpoint}...")
        self.predictor = Sam3VideoPredictor(
            checkpoint_path=sam3_checkpoint,
            has_presence_token=True,
            apply_temporal_disambiguation=True
        )

        print(f"[SAM3Recall] SAM3 loaded successfully")

    def recall_all_instances(
        self,
        video_path: str,
        category: str,
        session_id: str = "recall_session"
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Use SAM3 to segment all instances of the category in video.

        Args:
            video_path: Path to video directory (JPEGImages/video_id)
            category: Simple category name (e.g., "dog", "cow")
            session_id: Session identifier

        Returns:
            all_masks: Dict[frame_idx][obj_id] -> binary mask
        """
        print(f"\n[SAM3Recall] Recalling all instances of category: '{category}'")
        print(f"[SAM3Recall] Video path: {video_path}")

        # Start SAM3 session
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
                session_id=session_id
            )
        )
        print(f"[SAM3Recall] Session started: {session_id}")

        # Add text prompt on frame 0 (following SAM3 official usage)
        # Note: Text prompt applies to ALL frames during propagation
        # We don't check detection results on frame 0 - let propagate_in_video handle it
        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=category
            )
        )

        print(f"[SAM3Recall] Text prompt '{category}' added to inference state")
        print(f"[SAM3Recall] Propagating through video (detecting on every frame)...")

        # Propagate through entire video
        # SAM3 will detect '{category}' on EVERY frame (not just tracking from frame 0)
        all_masks = {}

        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            out_obj_ids_frame = outputs.get("out_obj_ids", [])
            out_masks = outputs.get("out_binary_masks", None)

            if out_masks is not None and len(out_obj_ids_frame) > 0:
                if frame_idx not in all_masks:
                    all_masks[frame_idx] = {}

                for i, obj_id in enumerate(out_obj_ids_frame):
                    all_masks[frame_idx][int(obj_id)] = out_masks[i]

        # Close session
        self.predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

        print(f"[SAM3Recall] Propagation complete: {len(all_masks)} frames with masks")

        # Report detected object IDs
        all_obj_ids = set()
        for frame_masks in all_masks.values():
            all_obj_ids.update(frame_masks.keys())

        if len(all_obj_ids) > 0:
            print(f"[SAM3Recall] ✓ Detected {len(all_obj_ids)} unique objects: {sorted(all_obj_ids)}")
        else:
            print(f"[SAM3Recall] ⚠ Warning: No objects detected for '{category}' in entire video")

        return all_masks


class VisualTubeBuilder:
    """Stage 2: Build Visual Tubes (spatio-temporal representations) for each candidate"""

    def __init__(self, num_keyframes: int = 8, temp_dir: str = "visual_tubes_temp", visualization_style: str = "outline",
                 sampling_strategy: str = "uniform", custom_frame_indices: List[int] = None,
                 qframe_config: Dict = None):
        """
        Args:
            num_keyframes: Number of frames to include in each Visual Tube (used for uniform sampling)
            temp_dir: Temporary directory to save annotated frames for MLLM
            visualization_style: Visualization style ("bbox", "outline", "glow", "mask")
            sampling_strategy: Sampling strategy ("uniform", "custom", or "qframe")
            custom_frame_indices: Custom frame indices (used when sampling_strategy="custom")
            qframe_config: Configuration for Q-Frame (used when sampling_strategy="qframe")
        """
        self.num_keyframes = num_keyframes
        self.temp_dir = temp_dir
        self.visualization_style = visualization_style
        self.sampling_strategy = sampling_strategy
        self.custom_frame_indices = custom_frame_indices if custom_frame_indices else []
        self.qframe_config = qframe_config if qframe_config else {}

        # Initialize Q-Frame selector if needed
        self.qframe_selector = None
        if sampling_strategy == "qframe":
            from mapp_sam3_demo import QFrameSelector
            clip_model_path = self.qframe_config.get('clip_model_path')
            device = self.qframe_config.get('device', 'cuda')

            if not clip_model_path:
                raise ValueError("Q-Frame sampling requires 'clip_model_path' in qframe_config")

            # Convert 'auto' to 'cuda' for Long-CLIP (it doesn't support 'auto')
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            print(f"[VisualTube] Initializing Q-Frame selector...")
            self.qframe_selector = QFrameSelector(clip_model_path, device, apply_rebalancing=False)

        os.makedirs(temp_dir, exist_ok=True)

        if sampling_strategy == "qframe":
            print(f"[VisualTube] Sampling strategy: qframe")
            print(f"[VisualTube] Will use Q-Frame to select keyframes dynamically")
            print(f"[VisualTube] Number of candidates: {num_keyframes}")
        elif sampling_strategy == "custom" and custom_frame_indices:
            print(f"[VisualTube] Sampling strategy: custom")
            print(f"[VisualTube] Custom frame indices: {custom_frame_indices}")
            print(f"[VisualTube] Number of frames: {len(custom_frame_indices)}")
        elif sampling_strategy == "all":
            print(f"[VisualTube] Sampling strategy: all (keep all frames)")
            print(f"[VisualTube] Warning: This may generate very large inputs for MLLM!")
        else:
            print(f"[VisualTube] Sampling strategy: uniform")
            print(f"[VisualTube] Configured with {num_keyframes} keyframes per tube")

        print(f"[VisualTube] Visualization style: {visualization_style}")
        print(f"[VisualTube] Temporary frames will be saved to {temp_dir}")

    def build_tubes(
        self,
        video_path: str,
        all_masks: Dict[int, Dict[int, np.ndarray]],
        referring_expression: str = None
    ) -> Dict[int, Dict[str, any]]:
        """
        Build Visual Tubes for each object ID.

        Args:
            video_path: Path to video directory (JPEGImages/video_id)
            all_masks: Dict[frame_idx][obj_id] -> binary mask
            referring_expression: Referring expression (required for Q-Frame sampling)

        Returns:
            visual_tubes: Dict[obj_id] -> {
                'frame_arrays': List of annotated frames (numpy arrays),
                'frame_paths': List of paths to saved annotated frames,
                'frame_indices': List of real frame indices in original video
            }
        """
        # Load all video frames if using Q-Frame
        video_frames = None
        if self.sampling_strategy == "qframe":
            if not referring_expression:
                raise ValueError("Q-Frame sampling requires 'referring_expression'")

            print(f"[VisualTube] Loading video frames for Q-Frame selection...")
            video_frames = load_video_frames(video_path)
            print(f"[VisualTube] Loaded {len(video_frames)} frames")

        # Collect all object IDs
        all_obj_ids = set()
        for frame_masks in all_masks.values():
            all_obj_ids.update(frame_masks.keys())
        all_obj_ids = sorted(all_obj_ids)

        print(f"\n[VisualTube] Building tubes for {len(all_obj_ids)} objects")

        visual_tubes = {}

        for obj_id in all_obj_ids:
            # Find frames where this object appears
            frames_with_obj = []
            for frame_idx in sorted(all_masks.keys()):
                if obj_id in all_masks[frame_idx]:
                    frames_with_obj.append(frame_idx)

            if len(frames_with_obj) == 0:
                continue

            # Sample keyframes based on sampling strategy
            if self.sampling_strategy == "qframe":
                # Q-Frame sampling: use Q-Frame selector to choose keyframes
                print(f"[VisualTube] Running Q-Frame selection for Object {obj_id}...")
                qframe_indices = self.qframe_selector.select_keyframes(
                    video_frames,
                    referring_expression,
                    num_candidates=self.num_keyframes,
                    tau=self.qframe_config.get('tau', 0.8)
                )

                # Filter to only frames where this object appears
                selected_frames = [idx for idx in qframe_indices if idx in frames_with_obj]

                if len(selected_frames) == 0:
                    print(f"[VisualTube] Warning: Object {obj_id} has no frames in Q-Frame selection, using uniform fallback")
                    # Fallback to uniform sampling
                    if len(frames_with_obj) <= self.num_keyframes:
                        selected_frames = frames_with_obj
                    else:
                        step = len(frames_with_obj) // self.num_keyframes
                        selected_frames = [frames_with_obj[i * step] for i in range(self.num_keyframes)]

                print(f"[VisualTube] Object {obj_id}: Q-Frame selected {len(selected_frames)} frames: {selected_frames}")

            elif self.sampling_strategy == "custom" and self.custom_frame_indices:
                # Custom sampling: use absolute video frame indices
                selected_frames = []
                for abs_frame_idx in self.custom_frame_indices:
                    # Check if this absolute frame index has the object and is within video range
                    if abs_frame_idx in frames_with_obj:
                        selected_frames.append(abs_frame_idx)
                    else:
                        # Frame doesn't have the object or is out of range
                        if abs_frame_idx in all_masks.keys():
                            print(f"[VisualTube] Warning: Object {obj_id} not present in frame {abs_frame_idx}, skipping")
                        else:
                            print(f"[VisualTube] Warning: Frame {abs_frame_idx} out of video range, skipping")

                if len(selected_frames) == 0:
                    print(f"[VisualTube] Warning: Object {obj_id} has no valid custom frames, skipping")
                    continue
            elif self.sampling_strategy == "all":
                # Keep all frames where the object appears
                selected_frames = frames_with_obj
            else:
                # Uniform sampling: evenly spaced across object's lifespan
                if len(frames_with_obj) <= self.num_keyframes:
                    selected_frames = frames_with_obj
                else:
                    step = len(frames_with_obj) // self.num_keyframes
                    selected_frames = [frames_with_obj[i * step] for i in range(self.num_keyframes)]

            # Build annotated frames
            tube_frame_arrays = []
            tube_frame_paths = []
            tube_frame_indices = []  # Track real frame indices

            # Create temp directory for this object
            obj_temp_dir = os.path.join(self.temp_dir, f"obj_{obj_id}")
            os.makedirs(obj_temp_dir, exist_ok=True)

            for frame_idx in selected_frames:
                # Load frame from video path
                frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                if frame_idx >= len(frame_files):
                    continue

                frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))

                # Get mask for this object
                mask = all_masks[frame_idx][obj_id]

                # Apply visualization based on style
                if self.visualization_style == "mask":
                    # Mask mode: overlay binary mask on original frame
                    # Create a colored overlay for the mask region
                    annotated_frame = frame.copy()

                    # Create a lightweight semi-transparent overlay
                    # Lower alpha (0.25) ensures original appearance remains visible
                    # while still clearly marking the target region
                    mask_bool = mask > 0
                    overlay_color = np.array([0, 220, 0], dtype=np.uint8)  # Slightly softer green
                    alpha = 0.25  # Reduced from 0.5 to 0.25 for better visibility

                    # Apply overlay where mask is True
                    annotated_frame[mask_bool] = (
                        annotated_frame[mask_bool] * (1 - alpha) + overlay_color * alpha
                    ).astype(np.uint8)

                    tube_frame_arrays.append(annotated_frame)

                    # Save overlaid frame for MLLM (use real frame index in filename)
                    temp_path = os.path.join(obj_temp_dir, f"frame_{frame_idx:05d}.jpg")
                    Image.fromarray(annotated_frame).save(temp_path)
                    tube_frame_paths.append(temp_path)
                    tube_frame_indices.append(frame_idx)

                elif self.visualization_style == "bbox":
                    annotated_frame = self._draw_bbox_from_mask(frame, mask)
                    tube_frame_arrays.append(annotated_frame)

                    temp_path = os.path.join(obj_temp_dir, f"frame_{frame_idx:05d}.jpg")
                    Image.fromarray(annotated_frame).save(temp_path)
                    tube_frame_paths.append(temp_path)
                    tube_frame_indices.append(frame_idx)

                elif self.visualization_style == "glow":
                    annotated_frame = self._draw_glow_from_mask(frame, mask)
                    tube_frame_arrays.append(annotated_frame)

                    temp_path = os.path.join(obj_temp_dir, f"frame_{frame_idx:05d}.jpg")
                    Image.fromarray(annotated_frame).save(temp_path)
                    tube_frame_paths.append(temp_path)
                    tube_frame_indices.append(frame_idx)

                else:  # Default to "outline"
                    annotated_frame = self._draw_outline_from_mask(
                        frame, mask,
                        rainbow=True,      # Rainbow gradient
                        thickness=2,       # Thin line
                        opacity=1.0        # Full opacity
                    )
                    tube_frame_arrays.append(annotated_frame)

                    temp_path = os.path.join(obj_temp_dir, f"frame_{frame_idx:05d}.jpg")
                    Image.fromarray(annotated_frame).save(temp_path)
                    tube_frame_paths.append(temp_path)
                    tube_frame_indices.append(frame_idx)

            visual_tubes[obj_id] = {
                'frame_arrays': tube_frame_arrays,
                'frame_paths': tube_frame_paths,
                'frame_indices': tube_frame_indices  # Add real frame indices
            }
            print(f"[VisualTube] Object {obj_id}: {len(tube_frame_arrays)} frames (lifespan: {frames_with_obj[0]}-{frames_with_obj[-1]}), indices: {tube_frame_indices}")

        return visual_tubes

    def _get_rainbow_color(self, progress: float) -> tuple:
        """
        根据轮廓进度(0.0 - 1.0)生成彩虹色 (BGR格式)。
        用于实现类似 HSV 渐变的视觉效果。
        """
        hue = int(179 * progress)  # OpenCV Hue range: 0-179
        # Saturation=255, Value=255 保证颜色最鲜艳
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr_color)

    def _draw_outline_from_mask(self,
                                frame: np.ndarray,
                                mask: np.ndarray,
                                color: tuple = (255, 0, 0),
                                thickness: int = 2,
                                rainbow: bool = False,
                                opacity: float = 1.0) -> np.ndarray:
        """
        绘制 Mask 轮廓，支持彩虹渐变、自定义粗细和透明度。
        使用形态学闭运算和最大连通分量处理遮挡导致的离散闭包问题。

        Args:
            frame: 原始图像 (H, W, 3)
            mask: 二值掩码 (H, W)
            color: 单色模式下的颜色 BGR (默认红色), 仅在 rainbow=False 时生效
            thickness: 线条粗细 (建议 1-5)
            rainbow: 是否开启彩虹渐变模式
            opacity: 轮廓透明度 (0.0 - 1.0)

        Returns:
            绘制好轮廓的图像
        """
        # 1. 确保图像格式正确
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        # 复制一份用于绘制轮廓，方便后续做透明度混合
        canvas = frame.copy()

        # 2. 预处理 Mask：仅在有离散闭包时才进行智能连接
        mask_uint8 = (mask > 0).astype(np.uint8)

        # 提取所有离散闭包（连通分量）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        # 只有在存在多个离散闭包时才进行连接处理
        if num_labels > 2:  # 0 是背景，1+ 是前景，大于2说明有多个离散闭包
            from scipy.spatial.distance import cdist

            # 为每个连通分量提取边界框和边界点
            component_info = []
            for label_id in range(1, num_labels):  # 跳过背景 label=0
                component_mask = (labels == label_id).astype(np.uint8)
                if int(cv2.__version__[0]) > 3:
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    _, contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    boundary_points = largest_contour[:, 0, :]  # (N, 2)

                    component_info.append({
                        'boundary_points': boundary_points,
                        'centroid': centroids[label_id]
                    })
                else:
                    component_info.append(None)

            # 使用最小生成树连接所有组件
            num_components = len(component_info)
            connected = [False] * num_components
            connected[0] = True  # 从第一个组件开始

            mask_connected = mask_uint8.copy()

            for _ in range(num_components - 1):
                min_dist = float('inf')
                best_connection = None

                # 找到已连接组件到未连接组件的最佳连接
                for i in range(num_components):
                    if not connected[i] or component_info[i] is None:
                        continue
                    for j in range(num_components):
                        if connected[j] or component_info[j] is None:
                            continue

                        # 判断两个闭包的空间关系
                        info_i = component_info[i]
                        info_j = component_info[j]

                        centroid_i = info_i['centroid']
                        centroid_j = info_j['centroid']

                        dx = abs(centroid_j[0] - centroid_i[0])  # 水平距离
                        dy = abs(centroid_j[1] - centroid_i[1])  # 垂直距离

                        boundary_i = info_i['boundary_points']
                        boundary_j = info_j['boundary_points']

                        # 根据空间关系选择连接策略
                        if dx > dy:
                            # 左右分布：连接上边到上边，下边到下边
                            # 找上边点（y 坐标最小）
                            top_i = boundary_i[np.argmin(boundary_i[:, 1])]
                            top_j = boundary_j[np.argmin(boundary_j[:, 1])]
                            # 找下边点（y 坐标最大）
                            bottom_i = boundary_i[np.argmax(boundary_i[:, 1])]
                            bottom_j = boundary_j[np.argmax(boundary_j[:, 1])]

                            # 计算两组连接的距离
                            dist_top = np.linalg.norm(top_i - top_j)
                            dist_bottom = np.linalg.norm(bottom_i - bottom_j)
                            current_min_dist = (dist_top + dist_bottom) / 2  # 平均距离

                            connection_points = [(top_i, top_j), (bottom_i, bottom_j)]
                        else:
                            # 上下分布：连接左边到左边，右边到右边
                            # 找左边点（x 坐标最小）
                            left_i = boundary_i[np.argmin(boundary_i[:, 0])]
                            left_j = boundary_j[np.argmin(boundary_j[:, 0])]
                            # 找右边点（x 坐标最大）
                            right_i = boundary_i[np.argmax(boundary_i[:, 0])]
                            right_j = boundary_j[np.argmax(boundary_j[:, 0])]

                            # 计算两组连接的距离
                            dist_left = np.linalg.norm(left_i - left_j)
                            dist_right = np.linalg.norm(right_i - right_j)
                            current_min_dist = (dist_left + dist_right) / 2  # 平均距离

                            connection_points = [(left_i, left_j), (right_i, right_j)]

                        if current_min_dist < min_dist:
                            min_dist = current_min_dist
                            best_connection = (i, j, connection_points)

                if best_connection:
                    i, j, connection_points = best_connection
                    connected[j] = True

                    # 画两条连接线
                    for pt1, pt2 in connection_points:
                        cv2.line(mask_connected,
                                tuple(pt1.astype(int)),
                                tuple(pt2.astype(int)),
                                1, thickness=2)

            mask_for_contour = mask_connected
        else:
            # 只有一个连通分量（或没有），直接使用原始 mask，不做任何连接处理
            mask_for_contour = mask_uint8

        # 3. 提取轮廓
        # 使用 RETR_EXTERNAL 只提取每个连通分量的最外层轮廓
        # 使用 CHAIN_APPROX_NONE 获取所有点以保证渐变平滑
        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return frame

        # 4. 绘制逻辑（保留所有轮廓，包括离散的闭包）
        for contour in contours:
            # 过滤极小的噪点
            if len(contour) < 5:
                continue

            if rainbow:
                # === 彩虹模式 ===
                # 将轮廓点形状从 (N, 1, 2) 转换为 (N, 2)
                pts = contour[:, 0, :]
                num_pts = len(pts)

                # 逐段绘制线条
                for i in range(num_pts):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % num_pts])  # 连接下一个点，最后一点连接起点

                    # 计算当前段颜色
                    color_rainbow = self._get_rainbow_color(i / num_pts)

                    # 使用 cv2.LINE_AA 抗锯齿，让细线条看起来更顺滑
                    cv2.line(canvas, pt1, pt2, color_rainbow, thickness=thickness, lineType=cv2.LINE_AA)
            else:
                # === 单色模式 ===
                cv2.drawContours(canvas, [contour], -1, color, thickness=thickness, lineType=cv2.LINE_AA)

        # 5. 处理透明度 (Blending)
        if opacity < 1.0:
            return cv2.addWeighted(canvas, opacity, frame, 1 - opacity, 0)

        return canvas

    def _draw_bbox_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        绘制 Mask 的红色边界框。

        Args:
            frame: 原始图像 (H, W, 3)
            mask: 二值掩码 (H, W)

        Returns:
            绘制好边界框的图像
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        canvas = frame.copy()
        mask_uint8 = (mask > 0).astype(np.uint8)

        # 找到mask的边界框
        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 获取最大轮廓的边界框
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 画红色边界框
            box_color = (0, 0, 255)  # BGR: Red
            box_width = 4
            cv2.rectangle(canvas, (x, y), (x+w, y+h), box_color, box_width)

        return canvas

    # def _draw_glow_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    #     """
    #     绘制 Mask 的发光轮廓效果（黄色glow）。
    #
    #     Args:
    #         frame: 原始图像 (H, W, 3)
    #         mask: 二值掩码 (H, W)
    #
    #     Returns:
    #         绘制好发光轮廓的图像
    #     """
    #     if frame.dtype != np.uint8:
    #         frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    #
    #     result = frame.copy()
    #     mask_uint8 = (mask > 0).astype(np.uint8)
    #
    #     # 提取轮廓
    #     if int(cv2.__version__[0]) > 3:
    #         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     else:
    #         _, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     if not contours:
    #         return frame
    #
    #     # Glow参数
    #     glow_color_bgr = (0, 207, 255)  # BGR: Yellow (#FCF200 -> RGB(252, 242, 0) -> BGR(0, 242, 252))
    #     # 正确的黄色应该是 BGR(0, 242, 252) 或者更鲜艳的 (0, 255, 255)
    #     glow_color_bgr = (0, 255, 255)  # 纯黄色 BGR
    #     glow_thickness = 1
    #     glow_radius = 15
    #     glow_intensity = 1.2  # 增强发光强度
    #
    #     # 创建单独的glow层（黑色背景）
    #     h, w = frame.shape[:2]
    #     glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
    #
    #     # 在glow层上绘制核心轮廓
    #     for contour in contours:
    #         if len(contour) < 5:
    #             continue
    #         cv2.drawContours(glow_layer, [contour], -1, glow_color_bgr, thickness=glow_thickness, lineType=cv2.LINE_AA)
    #
    #     # 应用高斯模糊创建glow效果
    #     glow_blurred = cv2.GaussianBlur(glow_layer, (glow_radius*2+1, glow_radius*2+1), 0)
    #
    #     # 将glow层叠加到原图上（使用加法混合，模拟发光）
    #     # 转换为float进行计算
    #     result_float = result.astype(float)
    #     glow_float = glow_blurred.astype(float) * glow_intensity
    #
    #     # 加法混合（发光效果）
    #     result_float = result_float + glow_float
    #     result = np.clip(result_float, 0, 255).astype(np.uint8)
    #
    #     # 在最上层再画一遍清晰的核心轮廓
    #     for contour in contours:
    #         if len(contour) < 5:
    #             continue
    #         cv2.drawContours(result, [contour], -1, glow_color_bgr, thickness=2, lineType=cv2.LINE_AA)
    #
    #     return result
    def _draw_glow_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        绘制 Mask 的强烈外发光轮廓效果（黄色 Glow），仿照目标参考图。

        Args:
            frame: 原始图像 (H, W, 3), uint8 类型
            mask: 二值掩码 (H, W), 非0表示目标区域

        Returns:
            绘制好发光轮廓的图像
        """
        # 1. 基础数据准备
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        h, w = frame.shape[:2]
        # 确保 mask 是标准的 0-255 uint8 图像
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 提取轮廓用于最后的描边
        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return frame

        # ================== 参数设置 ==================
        # 颜色: 参考图 UI 中的 Hex #FCF200 -> RGB(252, 242, 0) -> BGR(0, 242, 252)
        glow_color_bgr = (0, 242, 252)

        # 控制发光向外扩散的范围（越大越宽）
        dilation_radius = 15

        # 控制发光的柔和程度（越大越模糊）
        blur_sigma = 20
        blur_ksize = int(blur_sigma * 3) | 1  # 自动计算核大小，确保是奇数

        # 控制发光的整体强度/不透明度 (0.0 - 2.0+), 1.0 为正常
        glow_intensity = 0.8
        # =============================================

        # --- 步骤 1: 创建扩散的发光 Alpha 通道 ---

        # 1.1 膨胀 Mask：确定发光向外延伸的边界
        # 使用椭圆核进行膨胀，效果更圆润
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        dilated_mask = cv2.dilate(mask_uint8, kernel)

        # 1.2 高斯模糊：使边界变得柔和，产生光晕效果
        blurred_mask = cv2.GaussianBlur(dilated_mask, (blur_ksize, blur_ksize), blur_sigma)

        # --- 步骤 2: 制作“中空”的外发光 Mask ---

        # 将图像转换为 float 0.0-1.0 进行精确计算
        blurred_mask_f = blurred_mask.astype(float) / 255.0
        original_mask_f = mask_uint8.astype(float) / 255.0

        # 核心逻辑：从模糊的大 Mask 中减去原始的清晰 Mask。
        # 这样可以确保发光主要集中在物体*外部*，物体内部保持相对干净。
        # 使用 np.clip 确保数值不小于 0
        outer_glow_alpha = np.clip(blurred_mask_f - original_mask_f, 0.0, 1.0)

        # 增强 Alpha 通道的强度，使发光更明显
        outer_glow_alpha = np.clip(outer_glow_alpha * glow_intensity, 0.0, 1.0)

        # 将单通道 Alpha 扩展为三通道，以便与彩色图像计算
        alpha_3c = cv2.merge([outer_glow_alpha, outer_glow_alpha, outer_glow_alpha])

        # --- 步骤 3: 合成发光效果 (Alpha Blending) ---

        # 创建一个纯黄色的图像层
        solid_glow_layer = np.zeros_like(frame, dtype=float)
        solid_glow_layer[:] = glow_color_bgr

        frame_f = frame.astype(float)

        # 线性插值合成公式：Result = Original * (1 - Alpha) + GlowColor * Alpha
        # 这比单纯的加法混合 (Add) 看起来更像半透明的辉光
        result_f = frame_f * (1.0 - alpha_3c) + solid_glow_layer * alpha_3c

        result = np.clip(result_f, 0, 255).astype(np.uint8)

        # --- 步骤 4: 添加清晰的内边缘描边 ---
        # 如参考图所示，在柔和的光晕内部，物体的实际边缘有一条清晰的亮线
        for contour in contours:
            if len(contour) < 5:
                continue
            # thickness=2 保证线条足够清晰
            cv2.drawContours(result, [contour], -1, glow_color_bgr, thickness=2, lineType=cv2.LINE_AA)

        return result

class SemanticDiscriminator:
    """Stage 3: Use MLLM to score Visual Tubes and filter candidates"""

    def __init__(self, mllm_path: str, prompt_template_path: str = "configs/discrimination_prompt.txt", prompt_type: str = "cot", model_type: str = "qwen2.5"):
        """
        Args:
            mllm_path: Path to video-capable MLLM (Qwen2.5-VL, Qwen3-VL, or Qwen3-VL-Thinking)
            prompt_template_path: Path to discrimination prompt template file
            prompt_type: Type of prompt ("cot" for Chain of Thought, "three_level" for Yes/Partial/No)
            model_type: Type of model ("qwen2.5", "qwen3", or "qwen3-thinking")
        """
        from transformers import AutoProcessor

        self.model_type = model_type.lower()
        self.prompt_type = prompt_type
        self.prompt_template_path = prompt_template_path  # Store for potential reload

        print(f"[Discriminator] Loading {self.model_type.upper()} from {mllm_path}")

        # Load model based on type
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            from transformers import Qwen3VLForConditionalGeneration
            self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
                mllm_path,
                dtype="auto",
                device_map="auto",
            ).eval()
            self.process_vision_info = None  # Qwen3 doesn't need this
        elif self.model_type == "qwen2.5":
            from transformers import Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mllm_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            self.process_vision_info = process_vision_info
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'qwen2.5', 'qwen3', or 'qwen3-thinking'")

        self.processor = AutoProcessor.from_pretrained(
            mllm_path,
            trust_remote_code=(self.model_type == "qwen2.5")
        )

        # Load discrimination prompt template
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

        # Store bbox-specific prompt if available
        self.bbox_prompt_template = None
        bbox_prompt_path = prompt_template_path.replace('.txt', '_bbox.txt')
        if os.path.exists(bbox_prompt_path):
            with open(bbox_prompt_path, 'r', encoding='utf-8') as f:
                self.bbox_prompt_template = f.read()
            print(f"[Discriminator] Bbox prompt template loaded from {bbox_prompt_path}")
        else:
            print(f"[Discriminator] No bbox-specific prompt found, will use default for both")

        print(f"[Discriminator] MLLM loaded successfully ({self.model_type.upper()})")
        print(f"[Discriminator] Prompt template loaded from {prompt_template_path}")
        print(f"[Discriminator] Prompt type: {prompt_type}")

    def score_and_filter(
        self,
        visual_tubes: Dict[int, Dict[str, any]],
        referring_expression: str,
        num_targets: int,
        return_details: bool = False,
        visualization_style: str = None
    ) -> tuple:
        """
        Score each Visual Tube and return Top-K object IDs.

        Args:
            visual_tubes: Dict[obj_id] -> {'frame_arrays': [...], 'frame_paths': [...]}
            referring_expression: Original complex expression
            num_targets: Number of targets to keep
            return_details: If True, return detailed scores for each candidate
            visualization_style: Visualization style used ('bbox', 'mask', etc.) for prompt selection

        Returns:
            If return_details=False:
                selected_obj_ids: List of Top-K object IDs
            If return_details=True:
                (selected_obj_ids, detailed_scores)
                where detailed_scores is Dict[obj_id] -> {'score': float, 'prob_yes': float, ...}
        """
        print(f"\n[Discriminator] Scoring {len(visual_tubes)} candidates")
        print(f"[Discriminator] Target expression: '{referring_expression}'")
        print(f"[Discriminator] Will select Top-{num_targets} objects")

        # Select prompt template based on visualization style
        if visualization_style == 'bbox' and self.bbox_prompt_template is not None:
            print(f"[Discriminator] Using BBOX-specific prompt (color-aware)")
            current_prompt_template = self.bbox_prompt_template
        else:
            print(f"[Discriminator] Using default prompt (visualization style: {visualization_style})")
            current_prompt_template = self.prompt_template

        scores = {}
        detailed_scores = {}

        # Select scoring function based on prompt type
        if self.prompt_type == "three_level":
            score_func = lambda obj_id, paths, expr: self._score_tube_three_level(obj_id, paths, expr, current_prompt_template)
        else:  # Default to CoT
            score_func = lambda obj_id, paths, expr: self._score_tube_cot(obj_id, paths, expr, current_prompt_template)

        for obj_id, tube_data in visual_tubes.items():
            result = score_func(obj_id, tube_data['frame_paths'], referring_expression)

            # Handle both old format (float) and new format (dict)
            if isinstance(result, dict):
                scores[obj_id] = result['score']
                detailed_scores[obj_id] = result
            else:
                # Backward compatibility: if old _score_tube_cot returns float
                scores[obj_id] = result
                detailed_scores[obj_id] = {'score': result}

        # Sort by score and select Top-K
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        selected_ids = sorted_ids[:num_targets]

        print(f"\n[Discriminator] ===== Scoring Results =====")
        for obj_id in sorted_ids:
            status = "✓ SELECTED" if obj_id in selected_ids else "✗ FILTERED"
            print(f"  Object {obj_id}: {scores[obj_id]:.3f} {status}")
        print(f"[Discriminator] ===============================\n")

        if return_details:
            return selected_ids, detailed_scores
        else:
            return selected_ids

    def _score_tube_cot(
        self,
        obj_id: int,
        tube_frame_paths: List[str],
        referring_expression: str,
        prompt_template: str = None
    ) -> float:
        """
        Score a single Visual Tube using MLLM with Chain of Thought + Fine-grained Scoring (0-100).

        Args:
            obj_id: Object ID
            tube_frame_paths: List of paths to annotated frames (in temporal order)
            referring_expression: Target expression
            prompt_template: Prompt template to use (if None, use self.prompt_template)

        Returns:
            score: Confidence score (0-1, normalized from 0-100)
        """
        # Use provided template or default
        if prompt_template is None:
            prompt_template = self.prompt_template

        # Load prompt from template and format with referring expression
        prompt = prompt_template.format(referring_expression=referring_expression)

        # Prepare video-based input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tube_frame_paths,  # List of frame paths in temporal order
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs based on model type
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            # Qwen3-VL: Simplified API with do_sample_frames=False for image lists
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False  # Required for list of images
            )
            inputs = inputs.to(self.mllm.device)
        else:
            # Qwen2.5-VL: Requires process_vision_info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                messages, return_video_kwargs=True
            )

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(self.mllm.device)

        with torch.no_grad():
            if self.model_type == "qwen3-thinking":
                # Qwen3-VL-Thinking specific generation parameters
                outputs = self.mllm.generate(
                    **inputs,
                    max_new_tokens=40960,  # Large output for thinking process
                    do_sample=True,
                    top_p=0.95,
                    top_k=20,
                    temperature=1.0,
                    repetition_penalty=1.0,
                )
            elif self.model_type == "qwen3":
                # Qwen3-VL specific generation parameters
                outputs = self.mllm.generate(
                    **inputs,
                    max_new_tokens=21,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    temperature=0.7,
                    repetition_penalty=1.0,
                )
            else:
                # Qwen2.5-VL generation parameters
                outputs = self.mllm.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

        # Decode only the generated part (trim input)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        print(f"\n[Discriminator] Object {obj_id} Raw Response:\n{response}\n")

        # Parse JSON response to extract confidence_score
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                json_str = re.sub(r',(\\s*[}\\]])', r'\\1', json_match.group())
                result = json.loads(json_str)

                confidence_score = result.get("confidence_score", 0)
                visual_summary = result.get("visual_summary", "N/A")
                text_requirements = result.get("text_requirements", "N/A")
                alignment_analysis = result.get("alignment_analysis", "N/A")

                # Normalize score to 0-1 range
                score = confidence_score / 100.0

                print(f"[Discriminator] Object {obj_id}:")
                print(f"  Visual Summary: {visual_summary}")
                print(f"  Text Requirements: {text_requirements}")
                print(f"  Alignment Analysis: {alignment_analysis}")
                print(f"  Confidence Score: {confidence_score}/100 -> {score:.3f}")

                return score
            except json.JSONDecodeError as e:
                print(f"[Discriminator] Object {obj_id}: JSON parsing failed: {e}")
                print(f"[Discriminator] Attempting fallback parsing...")

        # Fallback: try to extract confidence_score from text
        score_match = re.search(r'confidence[_\s]*score["\s:]*(\d+)', response, re.IGNORECASE)
        if score_match:
            confidence_score = int(score_match.group(1))
            score = confidence_score / 100.0
            print(f"[Discriminator] Object {obj_id}: Extracted score from text: {confidence_score}/100 -> {score:.3f}")
            return score

        # Last resort fallback: keyword-based scoring
        response_lower = response.lower()
        if "definite match" in response_lower or "81-100" in response_lower:
            score = 0.9
        elif "plausible" in response_lower or "51-80" in response_lower:
            score = 0.65
        elif "same category" in response_lower or "21-50" in response_lower:
            score = 0.35
        else:
            score = 0.1

        print(f"[Discriminator] Object {obj_id}: Fallback keyword-based score: {score:.3f}")
        return score

    def _score_tube_three_level(
        self,
        obj_id: int,
        tube_frame_paths: List[str],
        referring_expression: str,
        prompt_template: str = None
    ) -> float:
        """
        Score a single Visual Tube using MLLM with Three-Level Scoring (Yes/Partial/No).
        Uses logits probability weighting instead of text parsing.

        Args:
            obj_id: Object ID
            tube_frame_paths: List of paths to annotated frames (in temporal order)
            referring_expression: Target expression
            prompt_template: Prompt template to use (if None, use self.prompt_template)

        Returns:
            score: Confidence score (0-1, computed from Yes/Partial/No probabilities)
        """
        # Use provided template or default
        if prompt_template is None:
            prompt_template = self.prompt_template

        # Load prompt from template and format with referring expression
        prompt = prompt_template.format(referring_expression=referring_expression)

        # Prepare video-based input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tube_frame_paths,  # List of frame paths in temporal order
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs based on model type
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            # Qwen3-VL: Simplified API with do_sample_frames=False for image lists
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False  # Required for list of images
            )
            inputs = inputs.to(self.mllm.device)
        else:
            # Qwen2.5-VL: Requires process_vision_info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                messages, return_video_kwargs=True
            )

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(self.mllm.device)

        # Get token IDs for "Yes", "Partial", "No"
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        partial_token_id = self.processor.tokenizer.encode("Partial", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        with torch.no_grad():
            outputs = self.mllm.generate(
                **inputs,
                max_new_tokens=5,  # Only generate one token
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False  # Use greedy decoding for consistent probability distribution
            )

        # Get logits for the first generated token
        first_token_logits = outputs.scores[0][0]  # Shape: (vocab_size,)

        # Extract logits for Yes, Partial, No tokens
        yes_logit = first_token_logits[yes_token_id].item()
        partial_logit = first_token_logits[partial_token_id].item()
        no_logit = first_token_logits[no_token_id].item()

        # Compute softmax probabilities over these three tokens
        import torch.nn.functional as F
        three_logits = torch.tensor([yes_logit, partial_logit, no_logit])
        three_probs = F.softmax(three_logits, dim=0)

        prob_yes = three_probs[0].item()
        prob_partial = three_probs[1].item()
        prob_no = three_probs[2].item()

        # Weighted score: Yes=1.0, Partial=0.5, No=0.0
        score = prob_yes * 1.0 + prob_partial * 0.5 + prob_no * 0.0

        # Also decode the actual response for logging
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs.sequences)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        print(f"[Discriminator] Object {obj_id}: Response='{response}', Probs=[Yes:{prob_yes:.3f}, Partial:{prob_partial:.3f}, No:{prob_no:.3f}], Score={score:.3f}")

        # Return detailed scores for TTRL data preparation
        return {
            'score': score,
            'prob_yes': prob_yes,
            'prob_partial': prob_partial,
            'prob_no': prob_no,
            'prediction': response
        }


class CRFSAM3Pipeline:
    """Complete CRF-SAM3 pipeline"""

    def __init__(
        self,
        mllm_path: str,
        sam3_checkpoint: str,
        num_tube_frames: int = 8,
        visualization_style: str = "outline",
        prompt_type: str = "cot",
        prompt_template_path: str = "configs/discrimination_prompt.txt",
        sampling_strategy: str = "uniform",
        custom_frame_indices: List[int] = None,
        qframe_config: Dict = None,
        model_type: str = "qwen2.5",
        concept_use_qframe: bool = False
    ):
        """
        Args:
            mllm_path: Path to MLLM
            sam3_checkpoint: Path to SAM3 checkpoint
            num_tube_frames: Frames per Visual Tube (used for uniform sampling)
            visualization_style: Visualization style ("bbox", "outline", "glow", "mask")
            prompt_type: Type of discrimination prompt ("cot" or "three_level")
            prompt_template_path: Path to prompt template file
            sampling_strategy: Sampling strategy ("uniform", "custom", or "qframe")
            custom_frame_indices: Custom frame indices (used when sampling_strategy="custom")
            qframe_config: Configuration for Q-Frame (used when sampling_strategy="qframe")
            model_type: MLLM type ("qwen2.5", "qwen3", or "qwen3-thinking")
            concept_use_qframe: Whether ConceptExtractor should use Q-Frame for keyframe selection
        """
        # CRITICAL: Store the original configured visualization_style to fix bbox persistence bug
        # When processing multiple samples sequentially, dynamic switching to bbox should always
        # restore to this original config, not to runtime captured state
        self._original_visualization_style = visualization_style

        self.concept_extractor = ConceptExtractor(
            mllm_path,
            model_type,
            use_qframe=concept_use_qframe,
            qframe_config=qframe_config
        )
        self.sam3_recall = SAM3ConceptRecall(sam3_checkpoint)
        self.tube_builder = VisualTubeBuilder(
            num_tube_frames,
            visualization_style=visualization_style,
            sampling_strategy=sampling_strategy,
            custom_frame_indices=custom_frame_indices,
            qframe_config=qframe_config
        )
        self.discriminator = SemanticDiscriminator(mllm_path, prompt_template_path, prompt_type, model_type)

        print("[CRFSAM3] Pipeline initialized")

    def predict(
        self,
        video_path: str,
        referring_expression: str,
        return_details: bool = False
    ) -> tuple:
        """
        Run CRF-SAM3 pipeline.

        Args:
            video_path: Path to video directory (JPEGImages/video_id)
            referring_expression: Complex referring expression
            return_details: If True, return detailed discrimination scores

        Returns:
            If return_details=False:
                (final_masks, visual_tubes)
            If return_details=True:
                (final_masks, visual_tubes, discrimination_details)
                where discrimination_details contains:
                    - 'category': extracted category
                    - 'num_targets': number of targets
                    - 'all_candidates': list of all candidate obj_ids
                    - 'selected_candidates': list of selected obj_ids
                    - 'scores': Dict[obj_id] -> {'score': float, 'prob_yes': float, ...}
        """
        print(f"\n{'='*60}")
        print(f"[CRFSAM3] Starting pipeline")
        print(f"[CRFSAM3] Expression: '{referring_expression}'")
        print(f"[CRFSAM3] Video path: {video_path}")
        print(f"{'='*60}\n")

        # CRITICAL: Detect if expression contains color attributes
        # FIXED: Previously captured runtime state causing bbox to persist across samples
        # Now always restore to the original configured style from __init__
        has_color = contains_color_attribute(referring_expression)
        if has_color:
            print(f"[CRFSAM3] ⚠ Color attribute detected in expression!")
            print(f"[CRFSAM3] Switching to 'bbox' visualization to preserve color information")
            print(f"[CRFSAM3] Original configured style: {self._original_visualization_style}")
            # Temporarily override visualization_style
            self.tube_builder.visualization_style = 'bbox'
        else:
            print(f"[CRFSAM3] No color attribute detected, using configured visualization style: {self.tube_builder.visualization_style}")

        # Stage 1: Concept Recall
        category, num_targets, category_reasoning = self.concept_extractor.extract_category(referring_expression, video_path)
        all_masks = self.sam3_recall.recall_all_instances(video_path, category)

        if len(all_masks) == 0:
            print("[CRFSAM3] ⚠ No objects recalled, returning empty result")
            if return_details:
                return {}, {}, {
                    'category': category,
                    'num_targets': num_targets,
                    'sub_conditions': category_reasoning.get("sub_conditions", []),
                    'final_rule_applied': category_reasoning.get("final_rule_applied", "N/A"),
                    'all_candidates': [],
                    'selected_candidates': [],
                    'scores': {}
                }
            else:
                return {}, {}

        # Stage 2: Visual Tube Construction
        visual_tubes = self.tube_builder.build_tubes(video_path, all_masks, referring_expression)

        # Get all candidate obj_ids
        all_candidate_ids = sorted(visual_tubes.keys())

        # Stage 3: Semantic Discrimination
        # Pass visualization_style to discriminator for prompt selection
        current_vis_style = self.tube_builder.visualization_style
        if return_details:
            selected_obj_ids, detailed_scores = self.discriminator.score_and_filter(
                visual_tubes, referring_expression, num_targets,
                return_details=True,
                visualization_style=current_vis_style
            )
        else:
            selected_obj_ids = self.discriminator.score_and_filter(
                visual_tubes, referring_expression, num_targets,
                return_details=False,
                visualization_style=current_vis_style
            )

        # Restore original visualization style if modified
        # FIXED: Always restore to the original configured style from __init__,
        # not to runtime captured state, to prevent bbox persistence across samples
        if has_color:
            self.tube_builder.visualization_style = self._original_visualization_style
            print(f"[CRFSAM3] Restored visualization style to original config: {self._original_visualization_style}")

        # Filter masks to keep only selected objects
        final_masks = {}
        for frame_idx, frame_masks in all_masks.items():
            final_masks[frame_idx] = {
                obj_id: mask
                for obj_id, mask in frame_masks.items()
                if obj_id in selected_obj_ids
            }

        print(f"\n[CRFSAM3] Pipeline complete!")
        print(f"[CRFSAM3] Selected {len(selected_obj_ids)} objects: {selected_obj_ids}")

        if return_details:
            discrimination_details = {
                'category': category,
                'num_targets': num_targets,
                'sub_conditions': category_reasoning.get("sub_conditions", []),
                'final_rule_applied': category_reasoning.get("final_rule_applied", "N/A"),
                'all_candidates': all_candidate_ids,
                'selected_candidates': selected_obj_ids,
                'scores': detailed_scores,
                'all_masks': all_masks  # Include all candidate masks for GT identification
            }
            return final_masks, visual_tubes, discrimination_details
        else:
            return final_masks, visual_tubes

    def save_visual_tubes(self, visual_tubes: Dict[int, Dict[str, any]], output_dir: str):
        """
        Save visual tubes to disk for visualization.

        Args:
            visual_tubes: Dict[obj_id] -> {'frame_arrays': [...], 'frame_paths': [...], 'frame_indices': [...]}
            output_dir: Directory to save tubes
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[CRFSAM3] Saving visual tubes to {output_dir}")

        for obj_id, tube_data in visual_tubes.items():
            obj_dir = os.path.join(output_dir, f"object_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)

            tube_frames = tube_data['frame_arrays']
            tube_frame_indices = tube_data.get('frame_indices', list(range(len(tube_frames))))  # Fallback to sequential if not available

            for real_frame_idx, frame in zip(tube_frame_indices, tube_frames):
                save_path = os.path.join(obj_dir, f"frame_{real_frame_idx:05d}.jpg")
                Image.fromarray(frame).save(save_path)

            print(f"[CRFSAM3] Saved {len(tube_frames)} frames for Object {obj_id} to {obj_dir} (indices: {tube_frame_indices})")

        print(f"[CRFSAM3] All visual tubes saved successfully")


def identify_gt_objects(
    all_masks_dict,
    discrimination_details,
    gt_masks_dict,
    iou_threshold=0.1
):
    """
    Identify GT objects by measuring individual IoU with ground truth.

    FIXED: Previous implementation had a critical bug where it accumulated masks
    in score order, causing the first object to "steal" GT regions from later objects.
    Now each object's IoU is computed independently.

    Args:
        all_masks_dict: Dict[frame_idx][obj_id] -> binary mask (all candidates from SAM3)
        discrimination_details: Discrimination details with candidate scores
        gt_masks_dict: Dict[frame_idx] -> ground truth mask (H, W) with object IDs
        iou_threshold: IoU threshold to mark as GT (default: 0.5)

    Returns:
        (gt_obj_ids, iou_per_object)
    """
    candidates = discrimination_details['all_candidates']

    if len(candidates) == 0 or len(gt_masks_dict) == 0:
        return [], {}

    gt_obj_ids = []
    iou_per_object = {}

    for obj_id in candidates:
        # Collect this object's masks (independent of other objects)
        obj_masks = {}
        for frame_idx in all_masks_dict.keys():
            if frame_idx in gt_masks_dict and obj_id in all_masks_dict[frame_idx]:
                obj_masks[frame_idx] = all_masks_dict[frame_idx][obj_id]

        # Compute IoU with GT (independently for each object)
        if len(obj_masks) > 0:
            iou = compute_j_score_sequence(obj_masks, gt_masks_dict)
            iou_per_object[obj_id] = iou

            # Mark as GT if IoU >= threshold
            if iou >= iou_threshold:
                gt_obj_ids.append(obj_id)
        else:
            iou_per_object[obj_id] = 0.0

    return gt_obj_ids, iou_per_object


def compute_j_score(pred_mask, gt_mask):
    """Compute J-score (IoU) between prediction and ground truth"""
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_j_score_sequence(pred_masks_dict, gt_masks_dict):
    """Compute average J-score over all frames"""
    j_scores = []

    for frame_idx in pred_masks_dict.keys():
        if frame_idx in gt_masks_dict:
            j = compute_j_score(pred_masks_dict[frame_idx], gt_masks_dict[frame_idx])
            j_scores.append(j)

    if len(j_scores) == 0:
        return 0.0

    return np.mean(j_scores)


def load_gt_masks(gt_masks_root, video_id, exp_id):
    """
    Load ground truth masks for a video expression

    Args:
        gt_masks_root: Root directory for GT masks (e.g., /path/to/gt_masks)
        video_id: Video ID
        exp_id: Expression ID

    Returns:
        Dict[frame_idx] -> mask (H, W) with object IDs
    """
    annotations_dir = os.path.join(gt_masks_root, video_id, str(exp_id))

    if not os.path.exists(annotations_dir):
        return {}

    mask_files = sorted(glob.glob(os.path.join(annotations_dir, "*.png")))
    if len(mask_files) == 0:
        return {}

    gt_masks = {}
    for mask_file in mask_files:
        frame_idx = int(os.path.basename(mask_file).replace('.png', ''))
        mask = np.array(Image.open(mask_file))
        gt_masks[frame_idx] = mask

    return gt_masks


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load video frames from directory"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    frame_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        frame_files.extend(glob.glob(os.path.join(video_path, ext)))

    frame_files.sort()

    frames = []
    for frame_file in frame_files:
        frame = np.array(Image.open(frame_file).convert("RGB"))
        frames.append(frame)

    return frames


def load_mevis_sample(mevis_root: str, video_id: str, exp_id: int):
    """
    Load a sample from MeViS dataset (following mapp_sam3_demo.py)

    Args:
        mevis_root: Path to MeViS dataset root (e.g., /path/to/mevis/valid_u)
        video_id: Video ID
        exp_id: Expression ID

    Returns:
        video_path: Path to video frames directory
        expression: Referring expression
        num_frames: Total number of frames
    """
    # Load meta data
    meta_file = os.path.join(mevis_root, "meta_expressions.json")
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    # Get expression
    video_meta = meta_data["videos"][video_id]
    exp_data = video_meta["expressions"][str(exp_id)]
    expression = exp_data["exp"]
    num_frames = len(video_meta["frames"])

    # Get video frames path
    video_path = os.path.join(mevis_root, "JPEGImages", video_id)

    print(f"[load_mevis_sample] Video: {video_id}")
    print(f"[load_mevis_sample] Expression {exp_id}: '{expression}'")
    print(f"[load_mevis_sample] Frames: {num_frames}")
    print(f"[load_mevis_sample] Path: {video_path}")

    return video_path, expression, num_frames



if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="CRF-SAM3 Demo - Single video inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file with all defaults
  python crf_sam3_demo.py --config configs/crf_sam3_config.yaml

  # Override video and expression
  python crf_sam3_demo.py --config configs/crf_sam3_config.yaml --video_path /path/to/video --expression "the dog running"

  # Override parameters
  python crf_sam3_demo.py --config configs/crf_sam3_config.yaml --num_tube_frames 12

  # Without config (manual parameters - requires all parameters)
  python crf_sam3_demo.py --video_path /path/to/video --expression "the dog" --mllm_path /path/to/mllm --sam3_checkpoint /path/to/sam3.pt
        """
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )

    # Required arguments (can be optional if using config with defaults)
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to video directory (JPEGImages/video_id)"
    )
    parser.add_argument(
        "--expression",
        type=str,
        help="Referring expression"
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="0",
        help="Expression ID (for output directory structure)"
    )

    # Optional overrides
    parser.add_argument("--mllm_path", type=str, help="Path to MLLM model")
    parser.add_argument("--sam3_checkpoint", type=str, help="Path to SAM3 checkpoint")
    parser.add_argument("--num_tube_frames", type=int, help="Number of frames per Visual Tube")
    parser.add_argument("--output_dir", type=str, help="Output directory for masks")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Extract parameters from config
        mllm_path = config['model']['mllm_path']
        sam3_checkpoint = config['model']['sam3_checkpoint']
        model_type = config['model'].get('model_type', 'qwen2.5')  # Default to qwen2.5
        num_tube_frames = config['pipeline']['num_tube_frames']

        # Extract visualization style and sampling config from visual_tube config
        visual_tube_config = config.get('visual_tube', {})
        visualization_style = visual_tube_config.get('visualization_style', 'outline')
        sampling_strategy = visual_tube_config.get('sampling_strategy', 'uniform')
        custom_frame_indices = visual_tube_config.get('custom_frame_indices', None)

        # Extract concept extraction config
        concept_extraction_config = config.get('concept_extraction', {})
        concept_use_qframe = concept_extraction_config.get('use_qframe', False)

        # Extract Q-Frame config if needed
        qframe_config = None
        if sampling_strategy == "qframe" or concept_use_qframe:
            qframe_config = visual_tube_config.get('qframe', {})
            # Add model paths if not specified
            if 'clip_model_path' not in qframe_config:
                qframe_config['clip_model_path'] = config.get('model', {}).get('clip_model_path')
            if 'device' not in qframe_config:
                qframe_config['device'] = config.get('pipeline', {}).get('device', 'cuda')
            if 'num_frames' not in qframe_config:
                qframe_config['num_frames'] = num_tube_frames  # Use same as tube frames

        # Extract discrimination prompt config
        discrimination_config = config.get('discrimination', {})
        prompt_type = discrimination_config.get('prompt_type', 'cot')

        # Determine prompt template path based on prompt_type
        prompt_templates = {
            'cot': 'configs/discrimination_prompt.txt',
            'three_level': 'configs/three_level_ttrl_cot.txt'  # TTRL-optimized prompt
        }
        prompt_template_path = prompt_templates.get(prompt_type, 'configs/discrimination_prompt.txt')

        # Get demo defaults
        demo_config = config.get('demo', {})
        mevis_root = demo_config.get('mevis_root')
        video_id = demo_config.get('video_id')
        exp_id = demo_config.get('exp_id', 0)
        output_dir = demo_config.get('output_dir', 'crf_sam3_demo_output')

        # Load from MeViS if mevis_root is provided
        if mevis_root and video_id is not None:
            video_path, expression, num_frames = load_mevis_sample(mevis_root, video_id, exp_id)
        else:
            # Fallback to direct paths (for backward compatibility)
            video_path = demo_config.get('video_path')
            expression = demo_config.get('expression')

        # Override with command-line arguments
        if args.mllm_path:
            mllm_path = args.mllm_path
        if args.sam3_checkpoint:
            sam3_checkpoint = args.sam3_checkpoint
        if args.num_tube_frames:
            num_tube_frames = args.num_tube_frames
        if args.video_path:
            video_path = args.video_path
        if args.expression:
            expression = args.expression
        if args.output_dir:
            output_dir = args.output_dir

        # Validate required parameters
        if not video_path:
            parser.error("--video_path is required (either in config demo.video_path or as command-line argument)")
        if not expression:
            parser.error("--expression is required (either in config demo.expression or as command-line argument)")
    else:
        parser.error("--config is required. Non-config mode is no longer supported.")

    # Extract GT masks root from config (if available)
    gt_masks_root = config.get('dataset', {}).get('gt_masks_root', '/mnt/csip-200/share/kongjiawei/Sa2VA/gt_masks')

    # Initialize pipeline
    print("="*60)
    print("CRF-SAM3 Demo")
    print("="*60)
    print(f"Model Type: {model_type.upper()}")
    print(f"MLLM: {mllm_path}")
    print(f"SAM3: {sam3_checkpoint}")
    print(f"Visual Tube Frames: {num_tube_frames}")
    print(f"Visualization Style: {visualization_style}")
    print(f"Sampling Strategy: {sampling_strategy}")
    if sampling_strategy == "custom" and custom_frame_indices:
        print(f"Custom Frame Indices: {custom_frame_indices}")
    elif sampling_strategy == "qframe":
        print(f"Q-Frame Config: {qframe_config}")
    print(f"Concept Extraction Q-Frame: {concept_use_qframe}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Prompt Template: {prompt_template_path}")
    print(f"Video: {video_path}")
    print(f"Expression: {expression}")
    print(f"Output: {output_dir}")
    print(f"GT Masks Root: {gt_masks_root}")
    print("="*60)

    pipeline = CRFSAM3Pipeline(
        mllm_path=mllm_path,
        sam3_checkpoint=sam3_checkpoint,
        num_tube_frames=num_tube_frames,
        visualization_style=visualization_style,
        prompt_type=prompt_type,
        prompt_template_path=prompt_template_path,
        sampling_strategy=sampling_strategy,
        custom_frame_indices=custom_frame_indices,
        qframe_config=qframe_config,
        model_type=model_type,
        concept_use_qframe=concept_use_qframe
    )

    # Run prediction with detailed scores
    masks_dict, visual_tubes, discrimination_details = pipeline.predict(
        video_path, expression, return_details=True
    )

    # Determine video_id and exp_id for output structure
    if 'video_id' not in locals() or video_id is None:
        video_id = os.path.basename(video_path)
    if 'exp_id' not in locals():
        exp_id = args.exp_id

    # Create output directory following eval script's expected structure: output_dir/video_id/exp_id/
    masks_output_dir = os.path.join(output_dir, video_id, str(exp_id))
    os.makedirs(masks_output_dir, exist_ok=True)

    # Save visual tubes for inspection (optional, in separate directory)
    visual_tubes_dir = os.path.join(output_dir, "visual_tubes", video_id, str(exp_id))
    pipeline.save_visual_tubes(visual_tubes, visual_tubes_dir)

    # Load GT masks and identify GT objects
    gt_masks_dict = load_gt_masks(gt_masks_root, video_id, exp_id)
    all_masks_dict = discrimination_details.get('all_masks', {})

    gt_obj_ids, iou_scores = identify_gt_objects(
        all_masks_dict,
        discrimination_details,
        gt_masks_dict,
        iou_threshold=0.1
    )

    # Build detailed scores
    candidates_scores = []
    for obj_id in discrimination_details['all_candidates']:
        score_info = discrimination_details['scores'].get(obj_id, {})
        candidates_scores.append({
            'obj_id': obj_id,
            'score': score_info.get('score', 0.0),
            'prob_yes': score_info.get('prob_yes', 0.0),
            'prob_partial': score_info.get('prob_partial', 0.0),
            'prob_no': score_info.get('prob_no', 0.0),
            'prediction': score_info.get('prediction', 'N/A'),
            'is_selected': obj_id in discrimination_details['selected_candidates'],
            'is_gt': obj_id in gt_obj_ids,
            'iou': iou_scores.get(obj_id, 0.0)
        })

    # Save detailed scores to JSON
    scores_data = {
        'video_id': video_id,
        'exp_id': exp_id,
        'expression': expression,
        'category': discrimination_details['category'],
        'num_targets': discrimination_details['num_targets'],
        'sub_conditions': discrimination_details.get('sub_conditions', []),
        'final_rule_applied': discrimination_details.get('final_rule_applied', 'N/A'),
        'num_candidates': len(discrimination_details['all_candidates']),
        'num_selected': len(discrimination_details['selected_candidates']),
        'num_gt_identified': len(gt_obj_ids),
        'gt_match': len(gt_obj_ids) == discrimination_details['num_targets'],
        'candidates': candidates_scores,
        'selected_obj_ids': discrimination_details['selected_candidates'],
        'gt_obj_ids': gt_obj_ids
    }

    scores_path = os.path.join(output_dir, video_id, str(exp_id), "scores.json")
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, 'w') as f:
        json.dump(scores_data, f, indent=2)

    print(f"\n[Demo] Saved detailed scores to {scores_path}")
    print(f"[Demo] GT objects identified: {len(gt_obj_ids)} (expected: {discrimination_details['num_targets']})")
    print(f"[Demo] GT match: {'✓' if scores_data['gt_match'] else '✗'}")

    # Save results
    if masks_dict:
        print(f"\nSaving masks to {masks_output_dir}...")
        for frame_idx in sorted(masks_dict.keys()):
            frame_masks = masks_dict[frame_idx]

            # Create instance mask (combine all objects)
            if len(frame_masks) == 0:
                continue

            first_mask = next(iter(frame_masks.values()))
            h, w = first_mask.shape
            instance_mask = np.zeros((h, w), dtype=np.uint8)

            for obj_id, mask in frame_masks.items():
                # Use modulo to ensure value stays within uint8 range (0-255)
                # Use bitwise OR to preserve all objects when masks overlap
                gray_value = ((obj_id + 1) * 50) % 256
                instance_mask = np.bitwise_or(instance_mask, (mask > 0).astype(np.uint8) * gray_value)

            # Save to video_id/exp_id/ directory
            mask_file = os.path.join(masks_output_dir, f"{frame_idx:05d}.png")
            Image.fromarray(instance_mask).save(mask_file)

        print(f"Saved {len(masks_dict)} frames to {masks_output_dir}")
        print(f"\nOutput structure for evaluation:")
        print(f"  {masks_output_dir}/")
        print(f"  └── 00000.png ~ {len(masks_dict)-1:05d}.png")
    else:
        print("No masks generated!")


