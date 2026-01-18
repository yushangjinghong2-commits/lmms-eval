#!/bin/bash
# OneCAT Integration Test Scripts
#
# These scripts demonstrate how to use OneCAT models with lmms-eval
# Make sure to download the OneCAT model before running

set -e  # Exit on error

# Change to lmms-eval directory
cd /home/xinjiezhang/data/lei/lmms-eval

# Define model paths
ONECAT_MODEL_PATH="/home/xinjiezhang/data/lei/lmms-eval/models/OneCAT-3B"
INFINITY_VAE_PATH="/home/xinjiezhang/data/lei/lmms-eval/models/infinity_vae/infinity_vae_d32reg.pth"

echo "========================================"
echo "OneCAT Integration Test Scripts"
echo "========================================"
echo ""
echo "Model paths:"
echo "  OneCAT-3B: $ONECAT_MODEL_PATH"
echo "  Infinity VAE: $INFINITY_VAE_PATH"
echo ""
echo "Available tests:"
echo "  1. Test onecat model (image understanding)"
echo "  2. Test onecat_visual_cot model (2-stage visual CoT)"
echo "  3. Run illusionbench icon shape test"
echo "  4. Run illusionbench icon shape visual CoT"
echo ""

# Check if model exists
if [ ! -d "$ONECAT_MODEL_PATH" ]; then
    echo "✗ OneCAT model not found at: $ONECAT_MODEL_PATH"
    echo ""
    echo "Please download the model first:"
    echo "  ./download_onecat.sh"
    echo ""
    exit 1
fi

# ============================================================
# Test 1: Basic onecat model test
# ============================================================
test_onecat() {
    echo "========================================"
    echo "Test 1: Testing onecat model"
    echo "========================================"

    python -m lmms_eval \
        --model onecat \
        --model_args pretrained=${ONECAT_MODEL_PATH} \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --limit 5 \
        --log_samples \
        --output_path ./logs/onecat_test/

    echo ""
    echo "✓ Test completed! Check logs at ./logs/onecat_test/"
}

# ============================================================
# Test 2: Visual CoT model test
# ============================================================
test_onecat_visual_cot() {
    echo "========================================"
    echo "Test 2: Testing onecat_visual_cot model"
    echo "========================================"

    if [ ! -f "$INFINITY_VAE_PATH" ]; then
        echo "✗ Infinity VAE not found at: $INFINITY_VAE_PATH"
        echo "  Visual CoT requires Infinity VAE for image generation"
        echo "  Please download it first:"
        echo "    ./download_onecat.sh"
        exit 1
    fi

    python -m lmms_eval \
        --model onecat_visual_cot \
        --model_args pretrained=${ONECAT_MODEL_PATH},vae_path=${INFINITY_VAE_PATH},save_intermediate=True \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --limit 5 \
        --log_samples \
        --output_path ./logs/onecat_visual_cot_test/

    echo ""
    echo "✓ Test completed! Check logs at ./logs/onecat_visual_cot_test/"
    echo "  Intermediate images saved for inspection"
}

# ============================================================
# Test 3: Full illusionbench icon shape test
# ============================================================
run_illusionbench_icon_shape() {
    echo "========================================"
    echo "Test 3: Full illusionbench icon shape test"
    echo "========================================"

    python -m lmms_eval \
        --model onecat \
        --model_args pretrained=${ONECAT_MODEL_PATH},max_new_tokens=64,do_sample=false \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/onecat_illusionbench_icon_shape/

    echo ""
    echo "✓ Evaluation completed! Check results at ./logs/onecat_illusionbench_icon_shape/"
}

# ============================================================
# Test 4: Full illusionbench icon shape visual CoT
# ============================================================
run_illusionbench_icon_shape_visual_cot() {
    echo "========================================"
    echo "Test 4: Full illusionbench icon shape visual CoT"
    echo "========================================"

    if [ ! -f "$INFINITY_VAE_PATH" ]; then
        echo "✗ Infinity VAE not found at: $INFINITY_VAE_PATH"
        echo "  Visual CoT requires Infinity VAE for image generation"
        echo "  Please download it first:"
        echo "    ./download_onecat.sh"
        exit 1
    fi

    python -m lmms_eval \
        --model onecat_visual_cot \
        --model_args pretrained=${ONECAT_MODEL_PATH},vae_path=${INFINITY_VAE_PATH},save_intermediate=True,stage1_cfg=1.5,stage1_top_k=2000,stage1_top_p=1.0 \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/onecat_visual_cot_illusionbench_icon_shape/

    echo ""
    echo "✓ Evaluation completed! Check results at ./logs/onecat_visual_cot_illusionbench_icon_shape/"
}

# ============================================================
# Main menu
# ============================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_number>"
    echo ""
    echo "Examples:"
    echo "  $0 1  # Run test 1 (basic onecat test)"
    echo "  $0 2  # Run test 2 (visual CoT test)"
    echo "  $0 3  # Run test 3 (full illusionbench icon shape)"
    echo "  $0 4  # Run test 4 (full illusionbench visual CoT)"
    exit 0
fi

case $1 in
    1)
        test_onecat
        ;;
    2)
        test_onecat_visual_cot
        ;;
    3)
        run_illusionbench_icon_shape
        ;;
    4)
        run_illusionbench_icon_shape_visual_cot
        ;;
    *)
        echo "Invalid test number. Choose 1, 2, 3, or 4"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
