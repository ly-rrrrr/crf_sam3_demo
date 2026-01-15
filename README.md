# CRF-SAM3 Demo 使用指南（简版）

## 概述

`crf_sam3_demo.py` 是 CRF-SAM3 的单视频推理演示脚本，采用“先召回、后过滤”的范式完成 RVOS 目标选择。

## 实验流程（无多帧拼接）

- **Stage 1: Concept Recall（概念召回）**  
  以指代表达式为输入，使用多模态语言模型解析出核心类别（category）与目标数量（num_targets）。  
  随后以类别词作为文本提示，驱动 SAM3 在全视频范围内进行候选实例召回，得到每个候选对象在时间维度上的掩码轨迹（masklets）。  
  该阶段的目标是“高召回、低漏检”，不强调精确筛选。

- **Stage 2: Visual Tube Construction（视觉管道构建）**  
  对每个候选对象的掩码轨迹进行关键帧采样（默认均匀采样），将掩码在原始帧上进行可视化标注（mask / bbox / outline / glow）。  
  由此形成候选对象的时序视觉管道（Visual Tube），保留跨帧的外观、动作与时空上下文。  
  注意：该版本不进行多帧拼接，仅输出逐帧可视化结果。

- **Stage 3: Semantic Discrimination（语义判别）**  
  将每条 Visual Tube 输入到 MLLM 进行语义判别与匹配评分。  
  得分来源可为结构化回答（如 confidence_score）或 logits 近似（Yes/Partial/No），最终将分数归一化到可比较尺度。  
  根据 Top-K 策略筛选（K = num_targets），保留高匹配度候选并输出最终分割掩码与评分记录。

## 运行方式（简版）

使用配置文件启动即可（示例：`configs/crf_sam3_config.yaml`）。核心配置包括：
`video_path`、`expression`、`output_dir`、`mllm_path`、`sam3_checkpoint`、`num_tube_frames`。

## 实际运行样本（无多帧拼接版本）

> 本样本来自 `crf_sam3_demo.py` 的真实输出，未使用 grid 拼接，仅保留逐帧 Visual Tube。

- **视频**: `mevis/valid_u/JPEGImages/2fbec459efc2`
- **表达式**: `white car move and turn left`
- **输出目录**: `crf_sam3_demo_output/2fbec459efc2/3`
- **scores.json**: `crf_sam3_demo_output/2fbec459efc2/3/scores.json`
- **visual_tubes 根目录**: `crf_sam3_demo_output/visual_tubes/2fbec459efc2/3/`

候选清单（每个候选对应一个 `object_x` 子目录）：
- `object_0`（frames=8, is_gt=False, score=1.4458e-05, iou=0.0001659）
- `object_1`（frames=3, is_gt=False, score=1.0723e-06, iou=0.0）
- `object_2`（frames=8, is_gt=True, score=0.9914, iou=0.9675）
- `object_3`（frames=8, is_gt=False, score=3.1052e-07, iou=0.0）
- `object_4`（frames=8, is_gt=False, score=2.2325e-05, iou=0.0）

## 输出结构（简版）

```
crf_sam3_demo_output/
├── <video_id>/<exp_id>/
│   ├── scores.json
│   ├── 00000.png
│   └── ...
└── visual_tubes/<video_id>/<exp_id>/
    ├── object_0/frame_*.jpg
    └── object_N/frame_*.jpg
```

## 常见问题（简版）

- **JSONDecodeError**：已在代码中加入清理逻辑，通常可自动修复。
- **候选分数同质**：可切换为 logits 评分或调整判别提示词。

## 相关文档

- `README_CRF_SAM3.md`
- `configs/crf_sam3_config.yaml`
