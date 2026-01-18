#!/bin/bash
# UniPic2 Integration Test Scripts
#
# These scripts demonstrate how to use UniPic2 models with lmms-eval
# Make sure to update the model paths before running

set -e  # Exit on error

# Activate the UniPic-2 environment
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

# Load model paths from configuration file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/model_paths.sh" ]; then
    source "$SCRIPT_DIR/model_paths.sh"
else
    echo "Error: model_paths.sh not found!"
    echo "Please run from the lmms-eval directory"
    exit 1
fi

echo "========================================"
echo "UniPic2 Integration Test Scripts"
echo "========================================"
echo ""
echo "Model paths loaded from cache:"
echo "  Qwen2.5-VL: $QWEN_MODEL_PATH"
echo "  UniPic2-Metaquery: $UNIPIC2_METAQUERY_PATH"
echo ""
echo "Available tests:"
echo "  1. Test unipic2 model (image understanding)"
echo "  2. Test unipic2_visual_cot model (2-stage visual CoT)"
echo "  3. Run illusionbench icon shape test"
echo "  4. Run illusionbench icon shape visual CoT"
echo ""

# ============================================================
# Test 1: Basic unipic2 model test
# ============================================================
test_unipic2() {
    echo "========================================"
    echo "Test 1: Testing unipic2 model"
    echo "========================================"

    accelerate launch -m lmms_eval \
        --model unipic2 \
        --model_args pretrained=${UNIPIC2_METAQUERY_PATH} \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --limit 5 \
        --log_samples \
        --output_path ./logs/unipic2_test/

    echo ""
    echo "✓ Test completed! Check logs at ./logs/unipic2_test/"
}

# ============================================================
# Test 2: Visual CoT model test
# ============================================================
test_unipic2_visual_cot() {
    echo "========================================"
    echo "Test 2: Testing unipic2_visual_cot model"
    echo "========================================"

    accelerate launch -m lmms_eval \
        --model unipic2_visual_cot \
        --model_args pretrained=${UNIPIC2_SD35M_PATH},qwen_model=${QWEN_MODEL_PATH},save_intermediate=True \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --limit 5 \
        --log_samples \
        --output_path ./logs/unipic2_visual_cot_test/

    echo ""
    echo "✓ Test completed! Check logs at ./logs/unipic2_visual_cot_test/"
    echo "  Intermediate images saved for inspection"
}

# ============================================================
# Test 3: Full illusionbench icon shape test
# ============================================================
run_illusionbench_icon_shape() {
    echo "========================================"
    echo "Test 3: Full illusionbench icon shape test"
    echo "========================================"

    accelerate launch -m lmms_eval \
        --model unipic2 \
        --model_args pretrained=${UNIPIC2_METAQUERY_PATH},max_new_tokens=64,temperature=0.0 \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/unipic2_illusionbench_icon_shape/

    echo ""
    echo "✓ Evaluation completed! Check results at ./logs/unipic2_illusionbench_icon_shape/"
}

# ============================================================
# Test 4: Full illusionbench icon shape visual CoT
# ============================================================
run_illusionbench_icon_shape_visual_cot() {
    echo "========================================"
    echo "Test 4: Full illusionbench icon shape visual CoT"
    echo "========================================"

    accelerate launch -m lmms_eval \
        --model unipic2_visual_cot \
        --model_args pretrained=${UNIPIC2_SD35M_PATH},qwen_model=${QWEN_MODEL_PATH},save_intermediate=True,stage1_num_inference_steps=50,stage1_guidance_scale=3.5 \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/unipic2_visual_cot_illusionbench_icon_shape/

    echo ""
    echo "✓ Evaluation completed! Check results at ./logs/unipic2_visual_cot_illusionbench_icon_shape/"
}

# ============================================================
# Main menu
# ============================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_number>"
    echo ""
    echo "Examples:"
    echo "  $0 1  # Run test 1 (basic unipic2 test)"
    echo "  $0 2  # Run test 2 (visual CoT test)"
    echo "  $0 3  # Run test 3 (full illusionbench icon shape)"
    echo "  $0 4  # Run test 4 (full illusionbench visual CoT)"
    exit 0
fi

case $1 in
    1)
        test_unipic2
        ;;
    2)
        test_unipic2_visual_cot
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
