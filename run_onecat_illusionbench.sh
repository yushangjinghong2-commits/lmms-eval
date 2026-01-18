#!/bin/bash
# Run illusionbench_arshia_test evaluation with OneCAT model

set -e

echo "========================================"
echo "OneCAT illusionbench_arshia_test Evaluation"
echo "========================================"
echo ""

# Change to lmms-eval directory
cd /home/xinjiezhang/data/lei/lmms-eval

# Check if OneCAT model exists
ONECAT_MODEL_PATH="/home/xinjiezhang/data/lei/lmms-eval/models/OneCAT-3B"

if [ ! -d "$ONECAT_MODEL_PATH" ]; then
    echo "✗ OneCAT-3B model not found at: $ONECAT_MODEL_PATH"
    echo ""
    echo "Please download the model first:"
    echo "  ./download_onecat.sh"
    echo ""
    echo "Or manually specify the model path if it's in a different location."
    exit 1
fi

echo "Task: illusionbench_arshia_test (6 subtasks)"
echo "Model: onecat"
echo "Model Path: $ONECAT_MODEL_PATH"
echo ""
echo "Subtasks included:"
echo "  - illusionbench_arshia_icon_shape_test"
echo "  - illusionbench_arshia_icon_scene_test"
echo "  - illusionbench_arshia_logo_shape_test"
echo "  - illusionbench_arshia_logo_scene_test"
echo "  - illusionbench_arshia_in_shape_test"
echo "  - illusionbench_arshia_in_scene_test"
echo ""
echo "Starting evaluation..."
echo "========================================"
echo ""

# Run evaluation
python -m lmms_eval \
    --model onecat \
    --model_args pretrained=$ONECAT_MODEL_PATH,max_new_tokens=64,do_sample=false \
    --tasks illusionbench_arshia_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/onecat_illusionbench_arshia_test/

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "========================================"
echo ""
echo "Results saved to: ./logs/onecat_illusionbench_arshia_test/"
echo ""
echo "To view results:"
echo "  cat ./logs/onecat_illusionbench_arshia_test/results.json"
