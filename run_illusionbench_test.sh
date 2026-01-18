#!/bin/bash
# Run illusionbench_arshia_test evaluation with unipic2 model

set -e

echo "========================================"
echo "illusionbench_arshia_test Evaluation"
echo "========================================"
echo ""

# Activate UniPic-2 environment
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

# Change to lmms-eval directory
cd /home/xinjiezhang/data/lei/lmms-eval

# Load model paths
source model_paths.sh

echo "Task: illusionbench_arshia_test (6 subtasks)"
echo "Model: unipic2"
echo "Model Path: $QWEN_MODEL_PATH"
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
accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=$QWEN_MODEL_PATH,max_new_tokens=64,temperature=0.0,do_sample=false \
    --tasks illusionbench_arshia_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/unipic2_illusionbench_arshia_test/

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "========================================"
echo ""
echo "Results saved to: ./logs/unipic2_illusionbench_arshia_test/"
echo ""
echo "To view results:"
echo "  cat ./logs/unipic2_illusionbench_arshia_test/results.json"
