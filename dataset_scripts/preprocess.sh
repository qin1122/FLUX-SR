#!/bin/bash

HR_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_HR"
LR_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_LR_x4"
PROMPT_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_prompt_short"
LR_FIX_SIZE_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_LR_x4_fixed"
HF_DATASET_DIR="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_dataset_x4_withprompt"

# downsample image
python generate_LR_image.py \
    --input "$HR_DIR" \
    --output "$LR_DIR" \
    --output_fix_size "$LR_FIX_SIZE_DIR"

# build hf dataset
python generate_dataset.py \
    --control_dir "$LR_DIR"\
    --target_dir "$HR_DIR" \
    --caption_dir "$PROMPT_DIR" \
    --output_dir "$HF_DATASET_DIR"