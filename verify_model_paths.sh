#!/bin/bash
# Quick verification script to check if model paths are valid

set -e

# Load model paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_paths.sh"

echo ""
echo "========================================"
echo "Model Path Verification"
echo "========================================"
echo ""

# Check Qwen2.5-VL model
echo "1. Checking Qwen2.5-VL-7B-Instruct..."
if [ -d "$QWEN_MODEL_PATH" ]; then
    echo "   ✓ Directory exists: $QWEN_MODEL_PATH"

    # Check for key files
    if [ -f "$QWEN_MODEL_PATH/config.json" ]; then
        echo "   ✓ config.json found"
    else
        echo "   ✗ config.json not found"
    fi

    if ls "$QWEN_MODEL_PATH"/*.safetensors >/dev/null 2>&1; then
        model_files=$(ls "$QWEN_MODEL_PATH"/*.safetensors | wc -l)
        echo "   ✓ Found $model_files safetensors file(s)"
    else
        echo "   ✗ No safetensors files found"
    fi
else
    echo "   ✗ Directory not found: $QWEN_MODEL_PATH"
fi

echo ""

# Check UniPic2-Metaquery model
echo "2. Checking UniPic2-Metaquery-9B..."
if [ -d "$UNIPIC2_METAQUERY_PATH" ]; then
    echo "   ✓ Directory exists: $UNIPIC2_METAQUERY_PATH"

    # Check for SD3.5M components
    components=("transformer" "vae" "scheduler" "conditioner")
    for component in "${components[@]}"; do
        if [ -d "$UNIPIC2_METAQUERY_PATH/$component" ]; then
            echo "   ✓ $component/ found"
        else
            echo "   ✗ $component/ not found"
        fi
    done
else
    echo "   ✗ Directory not found: $UNIPIC2_METAQUERY_PATH"
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo ""

# Test if we can import models using Python
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

python3 << 'PYEOF'
import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

qwen_path = os.path.expanduser(os.environ.get("QWEN_MODEL_PATH", ""))
unipic2_path = os.path.expanduser(os.environ.get("UNIPIC2_METAQUERY_PATH", ""))

success = True

print("Testing model loading (config only)...")
print()

# Test Qwen2.5-VL
try:
    from transformers import Qwen2_5_VLProcessor, AutoConfig
    print("1. Testing Qwen2.5-VL processor...")
    config = AutoConfig.from_pretrained(qwen_path)
    print(f"   ✓ Config loaded successfully")
    print(f"   ✓ Model type: {config.model_type}")
except Exception as e:
    print(f"   ✗ Failed to load Qwen2.5-VL: {e}")
    success = False

print()

# Test SD3.5M components
try:
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    print("2. Testing SD3.5M-Kontext components...")

    # Test transformer config
    transformer_config_path = os.path.join(unipic2_path, "transformer", "config.json")
    if os.path.exists(transformer_config_path):
        print(f"   ✓ Transformer config found")
    else:
        print(f"   ✗ Transformer config not found")
        success = False

    # Test VAE config
    vae_config_path = os.path.join(unipic2_path, "vae", "config.json")
    if os.path.exists(vae_config_path):
        print(f"   ✓ VAE config found")
    else:
        print(f"   ✗ VAE config not found")
        success = False

    # Test scheduler
    scheduler_config_path = os.path.join(unipic2_path, "scheduler", "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        print(f"   ✓ Scheduler config found")
    else:
        print(f"   ✗ Scheduler config not found")
        success = False

except Exception as e:
    print(f"   ✗ Failed to verify SD3.5M components: {e}")
    success = False

print()
print("=" * 40)
if success:
    print("✓ All model paths are valid and loadable!")
    print()
    print("You can now run:")
    print("  ./test_unipic2_integration.sh 1")
    sys.exit(0)
else:
    print("✗ Some models failed to load")
    print("Please check the error messages above")
    sys.exit(1)

PYEOF
