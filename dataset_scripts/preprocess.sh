#!/bin/bash

HR_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_train_HR"
LR_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_train_LR_x8"
PROMPT_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_train_prompt_short"
LR_FIX_SIZE_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_train_LR_x8_fixed"
HF_DATASET_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_train_dataset_x8_withprompt"

# downsample image
python dataset_scripts/generate_LR_image.py \
    --input "$HR_DIR" \
    --output "$LR_DIR" \
    --output_fix_size "$LR_FIX_SIZE_DIR"

# build hf dataset
python dataset_scripts/generate_dataset.py \
    --control_dir "$LR_DIR"\
    --target_dir "$HR_DIR" \
    --caption_dir "$PROMPT_DIR" \
    --output_dir "$HF_DATASET_DIR"

HR_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_valid_HR"
LR_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_valid_LR_x8"
LR_FIX_SIZE_DIR="/NASdata/dyw/FLUX-SR/datasets/DIV2K_valid_LR_x8_fixed"

# downsample image
python dataset_scripts/generate_LR_image.py \
    --input "$HR_DIR" \
    --output "$LR_DIR" \
    --output_fix_size "$LR_FIX_SIZE_DIR"