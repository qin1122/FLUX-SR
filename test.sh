#!/bin/bash

LORA_WEIGHT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/checkpoint"
IMAGE_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_LR_x4"
PROMPT_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_prompt_short"
GT_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_HR"
OUTPUT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/test"
STEP=100

python test_lora.py \
    --lora_weight_dir "$LORA_WEIGHT_DIR" \
    --image_dir "$IMAGE_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --step $STEP