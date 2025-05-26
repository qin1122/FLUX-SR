#!/bin/bash

HR_DIR="./datasets/DIV2K_train_HR"
LR_DIR="./datasets/DIV2K_train_LR_x"
HF_DATASET_DIR="./datasets/DIV2K_train_dataset_x"

# downsample image
python dataset_scr/generate_LR_image.py \
    --input $HR_DIR \
    --output $LR_DIR \
    --scale 16

# build hf dataset without prompt
python dataset_scr/generate_dataset.py \
    --control_dir $LR_DIR \
    --target_dir $HR_DIR \
    --output_dir $HF_DATASET_DIR \
    --scale 4 8 16


HR_DIR_v="./datasets/DIV2K_valid_HR"
LR_DIR_v="./datasets/DIV2K_valid_LR_x"

# downsample image
python dataset_scr/generate_LR_image.py \
    --input $HR_DIR_v \
    --output $LR_DIR_v \
    --scale 16