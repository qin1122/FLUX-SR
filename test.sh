#!/bin/bash

LORA_WEIGHT_DIR="results/train_with_prompt_x4/checkpoint"
IMAGE_DIR="datasets/DIV2K_valid_LR_x4"
PROMPT_DIR="datasets/DIV2K_valid_prompt_short"
GT_DIR="datasets/DIV2K_valid_HR"
OUTPUT_DIR="results/train_with_prompt_x4/test"
DEVICE="cuda:0"

for STEP in 900 1000 1100 1200; do
    echo "Running test for step $STEP..."
    
    python test.py \
        --lora_weight_dir "$LORA_WEIGHT_DIR" \
        --image_dir "$IMAGE_DIR" \
        --prompt_dir "$PROMPT_DIR" \
        --gt_dir "$GT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --step $STEP \
        --device $DEVICE
done