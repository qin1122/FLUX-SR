# Super Resolution by FLUX

## 🔧 Dependencies and Installation
- Python >= 3.11
- [PyTorch >= 2.4.0] (https://pytorch.org/)
- Diffusers == 0.33.0.dev0 (https://huggingface.co/docs/diffusers)

### Installation
1. Clone repo
    ```bash
    git clone https://github.com/qin1122/FLUX-SR.git
    cd FLUX-SR
    ```
2. Install dependent packages (use conda)
    ```bash
    conda create --name fluxsr_env python
    conda activate fluxsr_env
    pip install -r requirements.txt
    ```

## 🗂️ Datasets


## ⚙️ Training FLUX_SR
1. Log in to Hugging Face
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 Canny [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

2. Download DIV2K HR data
- [DIV2K Train data] (http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- [DIV2k Validation data] (http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)

3. Build FLUX_SR dataset
    ```bash
    ./dataset_scripts/preprocess.sh  # Change the args as needed
    ```

4. Fine-tuning FLUX with lora
    ```bash
    accelerate launch --config_file accelerate_config.yaml flux_lora_finetune_new.py \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Canny-dev" \
        --local_dataset_name="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_dataset_x4_withprompt" \
        --output_dir="/root/Homeworks/NLP/FLUX_SR/results/train_with_prompt_x4" \
        --mixed_precision="bf16" \
        --train_batch_size=1 \
        --rank=16 \
        --gradient_accumulation_steps=16 \
        --gradient_checkpointing \
        --learning_rate=1e-3 \
        --report_to="wandb" \
        --lr_scheduler="constant" \
        --lr_warmup_steps=60 \
        --max_train_steps=1250 \
        --checkpointing_steps=100 \
        --validation_image="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_LR_x4" \
        --validation_prompt="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_prompt_short" \
        --gt_image="/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_HR" \
        --valid_GPU_id="cuda:3" \
        --validation_steps=10000 \
        --offload\
        --seed="42" \
        --hub_token=""\
        --proportion_empty_prompts=0.0\
        --tracker_project_name="train_with_prompt_x4" \
        --use_8bit_adam
    ```

    The training script exposes additional CLI args that might be useful to experiment with:

    * `use_lora_bias`: When set, additionally trains the biases of the `lora_B` layer. 
    * `train_norm_layers`: When set, additionally trains the normalization scales. Takes care of saving and loading.
    * `lora_layers`: Specify the layers you want to apply LoRA to. If you specify "all-linear", all the linear layers will be LoRA-attached.

5. Test on DIV2K validation set
    ```bash
    ./test.sh  # Change the args as needed
    ```

## ⚡️ Inference

You can inference on your own image. Simply run:
```bash
./inference.sh  # Change the args as needed
```