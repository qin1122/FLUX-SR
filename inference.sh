#!/bin/bash

LORA_WEIGHT_DIR="./results/train_with_prompt_x4/checkpoint"
IMAGE_DIR="./datasets/DIV2K_valid_LR_x4/0803.png"
PROMPT_DIR="./datasets/DIV2K_valid_prompt_short/0803.txt"
GT_DIR="./datasets/DIV2K_valid_HR/0803.png"
OUTPUT_DIR="./results/train_with_prompt_x4/eval"
STEP=800
DEVICE="cuda:0"

python inference.py \
    --lora_weight_dir "$LORA_WEIGHT_DIR" \
    --image_dir "$IMAGE_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --step $STEP \
    --device $DEVICE