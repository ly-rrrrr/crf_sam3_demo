"""
Run CRF-SAM3 on MeViS dataset for evaluation

Usage:
    python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml
    python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml --max_videos 1
"""

import os
import argparse
import json
import yaml
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
from datetime import datetime

from crf_sam3_demo import CRFSAM3Pipeline, load_video_frames


def setup_logging(config):
    """Setup logging based on configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    handlers = [logging.StreamHandler()]

    if log_config.get('log_file'):
        log_dir = os.path.dirname(log_config['log_file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_config['log_file']))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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

    import glob
    mask_files = sorted(glob.glob(os.path.join(annotations_dir, "*.png")))
    if len(mask_files) == 0:
        return {}

    gt_masks = {}
    for mask_file in mask_files:
        frame_idx = int(os.path.basename(mask_file).replace('.png', ''))
        mask = np.array(Image.open(mask_file))
        gt_masks[frame_idx] = mask

    return gt_masks


def save_masks(masks_dict, output_dir, video_id, exp_id):
    """
    Save predicted masks to disk.

    Args:
        masks_dict: Dict[frame_idx][obj_id] -> binary mask
        output_dir: Root output directory
        video_id: Video ID
        exp_id: Expression ID
    """
    exp_output_dir = os.path.join(output_dir, video_id, str(exp_id))
    os.makedirs(exp_output_dir, exist_ok=True)

    # Collect all frames and object IDs
    all_frames = sorted(masks_dict.keys())
    all_obj_ids = set()
    for frame_masks in masks_dict.values():
        all_obj_ids.update(frame_masks.keys())
    all_obj_ids = sorted(all_obj_ids)

    print(f"  Saving {len(all_frames)} frames with {len(all_obj_ids)} objects")

    for frame_idx in all_frames:
        frame_masks = masks_dict[frame_idx]

        # Create instance mask (combine all objects)
        # Use SAME logic as crf_sam3_demo.py
        if len(frame_masks) == 0:
            continue

        # Get mask shape from first object
        first_mask = next(iter(frame_masks.values()))
        h, w = first_mask.shape
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        # Use same logic as crf_sam3_demo.py
        for obj_id, mask in frame_masks.items():
            # Use modulo to ensure value stays within uint8 range (0-255)
            # Use bitwise OR to preserve all objects when masks overlap
            gray_value = ((obj_id + 1) * 50) % 256
            instance_mask = np.bitwise_or(instance_mask, (mask > 0).astype(np.uint8) * gray_value)

        # Save as PNG
        mask_file = os.path.join(exp_output_dir, f"{frame_idx:05d}.png")
        Image.fromarray(instance_mask).save(mask_file)


def main():
    parser = argparse.ArgumentParser(
        description="Run CRF-SAM3 on MeViS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml

  # Override specific parameters
  python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml --max_videos 1 --num_tube_frames 12

  # Custom output directory
  python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml --output_dir my_results
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    # Optional overrides for config file parameters
    parser.add_argument("--mllm_path", type=str, help="Override MLLM path")
    parser.add_argument("--sam3_checkpoint", type=str, help="Override SAM3 checkpoint path")
    parser.add_argument("--data_root", type=str, help="Override dataset root")
    parser.add_argument("--split", type=str, choices=["valid", "valid_u", "train"], help="Override dataset split")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--max_videos", type=int, help="Override max videos")
    parser.add_argument("--max_expressions", type=int, help="Override max expressions per video")
    parser.add_argument("--num_tube_frames", type=int, help="Override number of frames per Visual Tube")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.mllm_path:
        config['model']['mllm_path'] = args.mllm_path
    if args.sam3_checkpoint:
        config['model']['sam3_checkpoint'] = args.sam3_checkpoint
    if args.data_root:
        config['dataset']['data_root'] = args.data_root
    if args.split:
        config['dataset']['split'] = args.split
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.max_videos is not None:
        config['dataset']['max_videos'] = args.max_videos
    if args.max_expressions is not None:
        config['dataset']['max_expressions'] = args.max_expressions
    if args.num_tube_frames:
        config['pipeline']['num_tube_frames'] = args.num_tube_frames

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("CRF-SAM3: Concept Recall & Spatio-Temporal Filtering")
    logger.info("="*60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract config values
    mllm_path = config['model']['mllm_path']
    sam3_checkpoint = config['model']['sam3_checkpoint']
    model_type = config['model'].get('model_type', 'qwen2.5')
    num_tube_frames = config['pipeline']['num_tube_frames']
    data_root = config['dataset']['data_root']
    split = config['dataset']['split']
    output_dir = config['output']['output_dir']
    max_videos = config['dataset'].get('max_videos')
    max_expressions = config['dataset'].get('max_expressions')
    save_visual_tubes = config['output'].get('save_visual_tubes', False)
    save_scores = config['output'].get('save_scores', True)
    use_tqdm = config['logging'].get('use_tqdm', True)

    # GT masks root directory
    gt_masks_root = config['dataset'].get('gt_masks_root', '/mnt/csip-200/share/kongjiawei/Sa2VA/gt_masks')

    # Discrimination parameters
    prompt_type = config.get('discrimination', {}).get('prompt_type', 'cot')

    # Determine prompt template path based on prompt_type
    prompt_templates = {
        'cot': 'configs/discrimination_prompt.txt',
        'three_level': 'configs/three_level_ttrl_cot.txt'  # TTRL-optimized prompt
    }
    prompt_template_path = prompt_templates.get(prompt_type, 'configs/discrimination_prompt.txt')

    visualization_style = config.get('visual_tube', {}).get('visualization_style', 'outline')
    sampling_strategy = config.get('visual_tube', {}).get('sampling_strategy', 'uniform')

    # Q-Frame configuration (only needed if sampling_strategy == "qframe")
    qframe_config = None
    if sampling_strategy == "qframe":
        clip_model_path = config['model'].get('clip_model_path')
        if clip_model_path:
            qframe_config = {
                'clip_model_path': clip_model_path,
                'device': config['pipeline'].get('device', 'cuda'),
                'tau': config.get('visual_tube', {}).get('qframe', {}).get('tau', 0.8)
            }

    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  MLLM: {mllm_path}")
    logger.info(f"  SAM3: {sam3_checkpoint}")
    logger.info(f"  Visual Tube Frames: {num_tube_frames}")
    logger.info(f"  Prompt Type: {prompt_type}")
    logger.info(f"  Prompt Template: {prompt_template_path}")
    logger.info(f"  Sampling Strategy: {sampling_strategy}")
    logger.info(f"  Visualization Style: {visualization_style}")
    logger.info(f"  Dataset: {data_root}/{split}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Max Videos: {max_videos if max_videos else 'All'}")
    logger.info(f"  Max Expressions: {max_expressions if max_expressions else 'All'}")

    # Initialize pipeline
    logger.info("\nInitializing CRF-SAM3 pipeline...")
    pipeline = CRFSAM3Pipeline(
        mllm_path=mllm_path,
        sam3_checkpoint=sam3_checkpoint,
        num_tube_frames=num_tube_frames,
        model_type=model_type,
        prompt_type=prompt_type,
        prompt_template_path=prompt_template_path,
        visualization_style=visualization_style,
        sampling_strategy=sampling_strategy,
        qframe_config=qframe_config
    )

    # Load dataset metadata
    meta_path = os.path.join(data_root, split, "meta_expressions.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    videos = meta['videos']
    logger.info(f"Found {len(videos)} videos in {split} split")

    # Process each video
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration to output directory
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_save_path}")

    processed_video_count = 0
    processed_exp_count = 0
    scores_log = [] if save_scores else None

    if use_tqdm:
        video_pbar = tqdm(videos.items(), desc="Videos", position=0)
    else:
        video_pbar = videos.items()

    for video_id, video_info in video_pbar:
        if max_videos is not None and processed_video_count >= max_videos:
            break

        if use_tqdm:
            video_pbar.set_description(f"Video {video_id}")

        # Get video path
        video_path = os.path.join(data_root, split, "JPEGImages", video_id)
        if not os.path.exists(video_path):
            logger.error(f"Video path does not exist: {video_path}")
            continue

        expressions = video_info['expressions']

        if use_tqdm:
            exp_pbar = tqdm(expressions.items(), desc="Expressions", position=1, leave=False)
        else:
            exp_pbar = expressions.items()

        exp_count = 0
        for exp_id, exp_info in exp_pbar:
            if max_expressions is not None and exp_count >= max_expressions:
                break

            referring_expression = exp_info['exp']
            if use_tqdm:
                exp_pbar.set_description(f"Exp {exp_id}: {referring_expression[:30]}...")

            try:
                # Run CRF-SAM3 with detailed scores
                masks_dict, visual_tubes, discrimination_details = pipeline.predict(
                    video_path, referring_expression, return_details=True
                )

                # Save results
                save_masks(masks_dict, output_dir, video_id, exp_id)

                # Save visual tubes if enabled
                if save_visual_tubes:
                    visual_tubes_dir = os.path.join(output_dir, "visual_tubes", video_id, str(exp_id))
                    pipeline.save_visual_tubes(visual_tubes, visual_tubes_dir)

                # Save scores if enabled
                if save_scores:
                    # Load GT masks and identify GT objects
                    gt_masks_dict = load_gt_masks(gt_masks_root, video_id, exp_id)
                    all_masks_dict = discrimination_details.get('all_masks', {})

                    gt_obj_ids, iou_scores = identify_gt_objects(
                        all_masks_dict,
                        discrimination_details,
                        gt_masks_dict,
                        iou_threshold=0.1
                    )

                    # Build candidate scores list
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

                    scores_log.append({
                        'video_id': video_id,
                        'exp_id': exp_id,
                        'expression': referring_expression,
                        'category': discrimination_details['category'],
                        'num_targets': discrimination_details['num_targets'],
                        'num_candidates': len(discrimination_details['all_candidates']),
                        'num_selected': len(discrimination_details['selected_candidates']),
                        'num_gt_identified': len(gt_obj_ids),
                        'gt_match': len(gt_obj_ids) == discrimination_details['num_targets'],
                        'candidates': candidates_scores,
                        'selected_obj_ids': discrimination_details['selected_candidates'],
                        'gt_obj_ids': gt_obj_ids
                    })

                processed_exp_count += 1
                exp_count += 1

            except Exception as e:
                logger.error(f"Error processing expression {exp_id} in video {video_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        processed_video_count += 1

    # Save scores log
    if save_scores and scores_log:
        scores_path = os.path.join(output_dir, "scores.json")
        with open(scores_path, 'w') as f:
            json.dump(scores_log, f, indent=2)
        logger.info(f"Saved scores to {scores_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  Videos processed: {processed_video_count}")
    logger.info(f"  Expressions processed: {processed_exp_count}")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
