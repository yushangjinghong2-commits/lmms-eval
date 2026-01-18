# UniPic2 Integration for lmms-eval

This integration allows you to use UniPic2 models with the lmms-eval framework to evaluate on illusionbench and other multimodal tasks.

## Overview

Two models have been integrated:

1. **unipic2**: Image understanding model based on Qwen2.5-VL (UniPic2-MetaQuery)
2. **unipic2_visual_cot**: Visual Chain-of-Thought model combining SD3.5M-Kontext (generation) and Qwen2.5-VL (understanding)

## Prerequisites

### 1. Clone UniPic-2 Repository

The UniPic2 models require custom modules from the UniPic-2 repository. Place it alongside the lmms-eval directory:

```bash
cd /home/xinjiezhang/data/lei
# UniPic-2 should already be cloned at /home/xinjiezhang/data/lei/UniPic/UniPic-2
```

### 2. Download Model Checkpoints

Download the required model checkpoints:

- **UniPic2-MetaQuery** (Qwen2.5-VL based): For image understanding
- **UniPic2-SD3.5M-Kontext**: For image generation (Visual CoT only)
- **Qwen2.5-VL-7B-Instruct**: Base model for understanding

Available at: https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd

### 3. Install Dependencies

Ensure you have the required dependencies in your environment:

```bash
# Activate the UniPic-2 environment
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

# Install additional dependencies if needed
pip install transformers diffusers accelerate torch pillow
```

## Usage

### Model 1: unipic2 (Image Understanding)

Use this model for standard image understanding tasks like illusionbench test tasks.

#### Example: Run illusionbench_arshia_icon_shape_test

```bash
accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=/path/to/UniPic2-MetaQuery \
    --tasks illusionbench_arshia_icon_shape_test \
    --batch_size 1 \
    --output_path ./logs/unipic2_illusionbench/
```

#### Model Arguments

- `pretrained`: Path to UniPic2-MetaQuery checkpoint (Qwen2.5-VL based)
- `max_new_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.0)
- `do_sample`: Whether to use sampling (default: False)
- `top_p`: Top-p sampling parameter (default: 1.0)
- `dtype`: Model dtype - "bfloat16", "float16", or "float32" (default: "bfloat16")
- `attn_implementation`: Attention implementation (default: "flash_attention_2")
- `continual_mode`: Enable response caching (default: True)
- `response_persistent_folder`: Cache directory (default: "./logs/unipic2_persistent_folder")

### Model 2: unipic2_visual_cot (Visual Chain-of-Thought)

Use this model for visual CoT tasks where auxiliary visualization helps answer questions.

#### Example: Run illusionbench_arshia_icon_shape_visual_cot

```bash
accelerate launch -m lmms_eval \
    --model unipic2_visual_cot \
    --model_args pretrained=/path/to/UniPic2-SD3.5M-Kontext,qwen_model=/path/to/Qwen2.5-VL-7B-Instruct \
    --tasks illusionbench_arshia_icon_shape_visual_cot \
    --batch_size 1 \
    --output_path ./logs/unipic2_visual_cot_illusionbench/
```

#### Model Arguments

**Required:**
- `pretrained`: Path to UniPic2-SD3.5M-Kontext checkpoint (for image generation)
- `qwen_model`: Path to Qwen2.5-VL-7B-Instruct checkpoint (for image understanding)

**Stage 1 (Image Generation):**
- `stage1_num_inference_steps`: Inference steps for generation (default: 50)
- `stage1_guidance_scale`: Guidance scale for generation (default: 3.5)
- `stage1_height`: Generated image height (default: 1024)
- `stage1_width`: Generated image width (default: 1024)

**Stage 2 (Visual Understanding):**
- `stage2_max_new_tokens`: Maximum tokens to generate (default: 512)
- `stage2_temperature`: Sampling temperature (default: 0.0)
- `stage2_do_sample`: Whether to use sampling (default: False)
- `stage2_top_p`: Top-p sampling parameter (default: 1.0)

**Other:**
- `dtype`: Model dtype - "bfloat16", "float16", or "float32" (default: "bfloat16")
- `seed`: Random seed (default: 0)
- `save_intermediate`: Save intermediate artifacts (default: False)
- `intermediate_dir`: Directory for intermediate artifacts
- `fail_gracefully`: Continue on errors (default: True)
- `generation_prompt_template`: Template for generation prompts

## Available Tasks

### illusionbench Tasks

The integration supports all illusionbench tasks:

**Test Tasks (using `unipic2`):**
- `illusionbench_arshia_icon_shape_test`
- `illusionbench_arshia_icon_scene_test`
- `illusionbench_arshia_logo_shape_test`
- `illusionbench_arshia_logo_scene_test`
- `illusionbench_arshia_in_shape_test`
- `illusionbench_arshia_in_scene_test`
- `illusionbench_arshia_test` (group task)

**Visual CoT Tasks (using `unipic2_visual_cot`):**
- `illusionbench_arshia_icon_shape_visual_cot`
- `illusionbench_arshia_icon_scene_visual_cot`
- `illusionbench_arshia_logo_shape_visual_cot`
- `illusionbench_arshia_logo_scene_visual_cot`
- `illusionbench_arshia_in_shape_visual_cot`
- `illusionbench_arshia_in_scene_visual_cot`
- `illusionbench_arshia_icon_visual_cot` (combined shape+scene)
- `illusionbench_arshia_logo_visual_cot` (combined shape+scene)
- `illusionbench_arshia_in_visual_cot` (combined shape+scene)

## How It Works

### unipic2 Model

1. Loads Qwen2.5-VL model (UniPic2-MetaQuery)
2. Processes input image and text prompt
3. Generates answer text directly

### unipic2_visual_cot Model

1. **Stage 1**: Uses SD3.5M-Kontext to generate an auxiliary visualization image based on the question
2. **Stage 2**: Uses Qwen2.5-VL to analyze both the original image and the auxiliary image to answer the question

This two-stage approach helps the model better understand complex visual reasoning tasks.

## Example Workflow

### 1. Standard Image Understanding

```bash
# Test on illusionbench icon shape recognition
accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=/path/to/UniPic2-MetaQuery,max_new_tokens=64,temperature=0.0 \
    --tasks illusionbench_arshia_icon_shape_test \
    --batch_size 1 \
    --output_path ./logs/unipic2_icon_shape/
```

### 2. Visual Chain-of-Thought

```bash
# Test on illusionbench with visual CoT
accelerate launch -m lmms_eval \
    --model unipic2_visual_cot \
    --model_args pretrained=/path/to/UniPic2-SD3.5M-Kontext,qwen_model=/path/to/Qwen2.5-VL-7B-Instruct,save_intermediate=True \
    --tasks illusionbench_arshia_icon_visual_cot \
    --batch_size 1 \
    --output_path ./logs/unipic2_visual_cot_icon/
```

When `save_intermediate=True`, the generated auxiliary images and metadata will be saved for inspection.

## Troubleshooting

### Import Error: UniPic2 modules not found

Ensure the UniPic-2 repository is in the correct location:
```
/home/xinjiezhang/data/lei/
├── lmms-eval/
└── UniPic/
    └── UniPic-2/
        ├── unipicv2/
        ├── scripts/
        └── ...
```

The integration automatically adds the UniPic-2 path to sys.path.

### CUDA Out of Memory

Try reducing batch size or using quantization:
- Reduce `stage1_height` and `stage1_width` for Visual CoT
- Use `dtype="float16"` instead of `bfloat16`
- Reduce `max_new_tokens`

### Flash Attention Not Available

If flash_attention_2 is not available, the model will fall back to standard attention automatically.

## Architecture Details

### unipic2.py

- Based on Qwen2.5-VL architecture
- Supports single-image understanding
- Uses the Qwen2.5-VL processor for input processing
- Implements continual mode for response caching

### unipic2_visual_cot.py

- Combines two models:
  - SD3.5M-Kontext for image generation
  - Qwen2.5-VL for image understanding
- Implements two-stage inference pipeline
- Supports saving intermediate artifacts for debugging
- Handles both single-image and dual-image inputs

## References

- [UniPic2 Paper](https://arxiv.org/abs/2509.04548)
- [UniPic2 GitHub](https://github.com/SkyworkAI/UniPic)
- [UniPic2 Models on HuggingFace](https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd)
- [lmms-eval Framework](https://github.com/EvolvingLMMs-Lab/lmms-eval)
