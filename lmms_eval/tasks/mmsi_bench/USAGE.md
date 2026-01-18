# MMSI-Bench Task Configuration

## Overview

MMSI-Bench (Multi-Modal Spatial Intelligence Benchmark) evaluation tasks for lmms-eval.

Dataset location: `/home/xinjiezhang/data/lei/datasets/mmsi_bench/`

## Task Groups

### 1. Standard Evaluation (mmsi_bench)
Evaluates all task categories with standard prompts.

**Usage:**
```bash
python -m lmms_eval \
  --model onecat \
  --model_args pretrained=/path/to/model \
  --tasks mmsi_bench \
  --batch_size 1 \
  --output_path ./logs/
```

**Includes:**
- `mmsi_msr` - Multi-Step Reasoning (100 samples)
- `mmsi_attribute_appr` - Attribute Appearance (100 samples)
- `mmsi_attribute_meas` - Attribute Measurement (100 samples)
- `mmsi_motion_cam` - Camera Motion (100 samples)
- `mmsi_motion_obj` - Object Motion (100 samples)

### 2. Visual CoT Evaluation (mmsi_bench_visual_cot)
Evaluates with visual chain-of-thought two-stage inference.

**Usage:**
```bash
python -m lmms_eval \
  --model onecat \
  --model_args pretrained=/path/to/model \
  --tasks mmsi_bench_visual_cot \
  --batch_size 1 \
  --output_path ./logs/
```

**Includes:**
- `mmsi_msr_visual_cot`
- `mmsi_attribute_appr_visual_cot`
- `mmsi_attribute_meas_visual_cot`
- `mmsi_motion_cam_visual_cot`
- `mmsi_motion_obj_visual_cot`

## Individual Task Evaluation

You can also run individual tasks:

```bash
# Standard
python -m lmms_eval --tasks mmsi_msr
python -m lmms_eval --tasks mmsi_attribute_appr
python -m lmms_eval --tasks mmsi_motion_cam

# Visual CoT
python -m lmms_eval --tasks mmsi_msr_visual_cot
python -m lmms_eval --tasks mmsi_attribute_appr_visual_cot
```

## Configuration Files

### Group Configurations
- `mmsi_bench.yaml` - Standard evaluation group
- `mmsi_bench_visual_cot.yaml` - Visual CoT evaluation group

### Individual Task Configurations
**Standard:**
- `mmsi_msr.yaml`
- `mmsi_attribute_appr.yaml`
- `mmsi_attribute_meas.yaml`
- `mmsi_motion_cam.yaml`
- `mmsi_motion_obj.yaml`

**Visual CoT:**
- `mmsi_msr_visual_cot.yaml`
- `mmsi_attribute_appr_visual_cot.yaml`
- `mmsi_attribute_meas_visual_cot.yaml`
- `mmsi_motion_cam_visual_cot.yaml`
- `mmsi_motion_obj_visual_cot.yaml`

## Dataset Files

Located in `/home/xinjiezhang/data/lei/datasets/mmsi_bench/`:
- `msr.parquet` - Multi-Step Reasoning
- `attribute_appr.parquet` - Attribute Appearance
- `attribute_meas.parquet` - Attribute Measurement
- `motion_cam.parquet` - Camera Motion
- `motion_obj.parquet` - Object Motion

## Key Differences

### Standard vs Visual CoT

**Standard Evaluation:**
- Uses `msr_doc_to_text` function
- Single-stage inference
- Task-specific prompts for reasoning guidance

**Visual CoT Evaluation:**
- Uses `msr_doc_to_text_with_gen_prompt` function
- Two-stage inference:
  1. Stage 1: Generate visual reasoning visualization
  2. Stage 2: Answer based on original images + visualization
- Includes `generation_prompt` for stage 1
- Modified prompts referencing the visualization

## Metrics

All tasks report accuracy for their respective categories:
- MSR (Multi-Step Reasoning)
- Attribute (Appr.) - Appearance attributes
- Attribute (Meas.) - Measurement attributes
- Motion (Cam.) - Camera motion
- Motion (Obj.) - Object motion

Average across all categories is also computed.
