# CRF-SAM3 Demo

CRF-SAM3（Concept Recall & Spatio-Temporal Filtering with SAM3）用于指代表达式视频目标分割（RVOS）的演示与评测。

## 流程概述
1. Concept Recall：MLLM 从表达式抽取类别/数量，SAM3 召回候选。
2. Visual Tube：采样多帧构建时空 tube，支持 bbox/outline/glow/mask 可视化。
3. Semantic Discrimination：MLLM 打分筛选候选（cot 或 three_level）。

## 仓库内容
- `crf_sam3_demo.py`：单视频 demo + 主 pipeline
- `crf_sam3_config.yaml`：默认配置
- `run_crf_sam3_dataset.py`：MeViS 数据集评测
- `run_crf_sam3_dataset_subexpr.py`：子表达式评测（依赖 `Visual-RFT/`）
- `category_num_targets_inference.txt`：类别/数量推断 prompt
- `three_level_ttrl_cot.txt`：三档判别 prompt
- `scores.json`：示例输出
- `README_CRF_SAM3_DEMO.md`：旧版说明（可能存在编码问题）

## 依赖与模型
- Python 3.8+
- 依赖（基于 import）：`torch`、`transformers`、`opencv-python`、`Pillow`、`numpy`、`pyyaml`、`tqdm`
- 需准备：SAM3 checkpoint、Qwen2.5/3 MLLM；启用 Q-Frame 时还需 CLIP 模型

## 快速开始
1. 修改 `crf_sam3_config.yaml`：
   - `model.mllm_path`、`model.sam3_checkpoint`、`model.clip_model_path`（可选）
   - `demo.mevis_root`、`demo.video_id`、`demo.exp_id`，或 `demo.video_path`/`demo.expression`
   - `pipeline.num_tube_frames`、`visual_tube.sampling_strategy`、`visual_tube.visualization_style`
   - `discrimination.prompt_type`（`cot`/`three_level`）
2. 运行：
   `python crf_sam3_demo.py --config crf_sam3_config.yaml`
   可覆盖参数：`--video_path`、`--expression`、`--num_tube_frames`、`--output_dir`
   注：当前脚本要求必须提供 `--config`。

## 示例（当前）
- 视频：`mevis/valid_u/JPEGImages/3dde46eaaf53`
- 表达式：`The two monkeys in a crouched position on the left without any movement.`
- 输出目录：`crf_sam3_demo_output/3dde46eaaf53/22`
- `scores.json`：`crf_sam3_demo_output/3dde46eaaf53/22/scores.json`
- visual_tubes：`crf_sam3_demo_output/visual_tubes/3dde46eaaf53/22/`

## 输出结构
```
crf_sam3_demo_output/
  <video_id>/<exp_id>/
    00000.png
    00001.png
    scores.json
  visual_tubes/<video_id>/<exp_id>/
    object_0/frame_*.jpg
    object_N/frame_*.jpg
```

## 数据集评测
- `python run_crf_sam3_dataset.py --config crf_sam3_config.yaml`
- 子表达式评测：`python run_crf_sam3_dataset_subexpr.py --config <your_config>`（需 `Visual-RFT/`）

## 常见问题
- Prompt 路径：代码默认读取 `configs/category_num_targets_inference.txt` 和 `configs/three_level_ttrl_cot.txt`。若你的 prompt 在仓库根目录，请新建 `configs/` 并移动文件，或修改代码中的路径。
- 编码异常：如提示模板出现乱码，建议确认文件以 UTF-8 保存。
