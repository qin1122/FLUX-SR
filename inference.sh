#!/bin/bash

# LORA_WEIGHT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/checkpoint-200"
LORA_WEIGHT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/checkpoint"
IMAGE_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_LR_x4/0801.png"
PROMPT_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_prompt_short/0801.txt"
GT_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_HR/0801.png"
OUTPUT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/eval"
STEP=100
# OUTPUT_DIR="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4/valid_200"

python inference.py \
    --lora_weight_dir "$LORA_WEIGHT_DIR" \
    --image_dir "$IMAGE_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --step $STEP