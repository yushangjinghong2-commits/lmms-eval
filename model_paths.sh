# UniPic2 Model Paths Configuration
# Auto-generated from HuggingFace cache

# Found models in HuggingFace cache:

# 1. Qwen2.5-VL-7B-Instruct (Base LLM for understanding)
QWEN_MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

# 2. UniPic2-Metaquery-9B (Unified model with SD3.5M-Kontext components)
# This model contains:
#   - transformer (SD3Transformer2DKontextModel)
#   - vae (AutoencoderKL)
#   - scheduler (FlowMatchEulerDiscreteScheduler)
#   - conditioner (StableDiffusion3Conditioner)
UNIPIC2_METAQUERY_PATH="$HOME/.cache/huggingface/hub/models--Skywork--UniPic2-Metaquery-9B/snapshots/37a2f17d28578b89d38aebd79515ba5610e75cad"

# For Visual CoT, use the same UniPic2-Metaquery model as it contains SD3.5M-Kontext
UNIPIC2_SD35M_PATH="$UNIPIC2_METAQUERY_PATH"

# Export for use in scripts
export QWEN_MODEL_PATH
export UNIPIC2_METAQUERY_PATH
export UNIPIC2_SD35M_PATH

echo "Model paths configured:"
echo "  QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
echo "  UNIPIC2_METAQUERY_PATH=$UNIPIC2_METAQUERY_PATH"
echo "  UNIPIC2_SD35M_PATH=$UNIPIC2_SD35M_PATH"
