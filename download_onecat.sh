#!/bin/bash
# Download and configure OneCAT-3B model for lmms-eval

set -e

echo "========================================"
echo "OneCAT Model Setup"
echo "========================================"
echo ""

# Create models directory
MODELS_DIR="/home/xinjiezhang/data/lei/lmms-eval/models"
mkdir -p "$MODELS_DIR"

echo "Models will be stored in: $MODELS_DIR"
echo ""

# Download OneCAT-3B from HuggingFace
echo "Downloading OneCAT-3B model..."
echo "This may take a while depending on your connection speed."
echo ""

cd "$MODELS_DIR"

# Use huggingface-cli to download
echo "Using huggingface-cli to download OneCAT-3B..."
huggingface-cli download onecat-ai/OneCAT-3B --local-dir OneCAT-3B --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ OneCAT-3B downloaded successfully to: $MODELS_DIR/OneCAT-3B"
else
    echo ""
    echo "✗ Download failed. Please check your internet connection and try again."
    echo ""
    echo "Alternative: Download manually from https://huggingface.co/onecat-ai/OneCAT-3B"
    exit 1
fi

# Download Infinity VAE tokenizer
echo ""
echo "Downloading Infinity VAE tokenizer..."
mkdir -p "$MODELS_DIR/infinity_vae"
cd "$MODELS_DIR/infinity_vae"

wget -O infinity_vae_d32reg.pth "https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32reg.pth?download=true"

if [ $? -eq 0 ]; then
    echo "✓ Infinity VAE downloaded successfully"
else
    echo "✗ VAE download failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Model paths:"
echo "  OneCAT-3B: $MODELS_DIR/OneCAT-3B"
echo "  Infinity VAE: $MODELS_DIR/infinity_vae/infinity_vae_d32reg.pth"
echo ""
echo "Next steps:"
echo "  1. Update model_paths.sh with OneCAT path"
echo "  2. Run: ./test_onecat_integration.sh"
