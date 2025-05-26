# Text Guided Super-Resolution with FLUX

## 🔧 Dependencies and Installation
- Python >= 3.11
- [PyTorch >= 2.4.0](https://pytorch.org/)
- [diffusers >= 0.33.0.dev0](https://huggingface.co/docs/diffusers)

### Installation
1. Clone repo

```bash
git clone https://github.com/qin1122/FLUX-SR.git
cd FLUX-SR
```
2. Install dependent packages (use conda)

```bash
conda create --name fluxsr_env
conda activate fluxsr_env
conda install python
pip install -r requirements.txt
```

## 🗂️ Datasets

Our datasets are build based on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.

- To obtain the low-resolution(LR) images, we use the degradation pipeline proposed by [RealESRGAN](https://github.com/xinntao/Real-ESRGAN). 

- We use OpenAI [GPT-4o](https://openai.com/index/hello-gpt-4o/) to generate prompts for each image.

Our preprocessed datasets can be downloaded [here](https://box.nju.edu.cn/d/b2895e81670b436c88d8/), download the dataset you need and place it in the ./datasets folder.

> datasets folder structure is as follows: 
>
> datasets/  -- Preprocessed datasets
> 
> datasets/DIV2K_train_HR -- 0001.png, 0002.png, ..., 0800.png, train HR images \
> datasets/DIV2K_train_LR_x4 -- train LR images, downscale factor x4 \
> datasets/DIV2K_train_LR_x8 -- train LR images, downscale factor x8 \
> datasets/DIV2K_train_LR_x16 -- train LR images, downscale factor x16 \
> datasets/DIV2K_train_prompt_short -- 0001.txt, 0002.txt, ..., 0800.txt, prompts for train images \
> datasets/DIV2K_train_dataset_x4_withprompt -- hf dataset, including train HR images, train LR images downscale factor x4 and prompts. \
> datasets/DIV2K_train_dataset_x4_withoutprompt -- hf dataset, including train HR images, train LR images downscale factor x4. \
> datasets/DIV2K_train_dataset_x8_withprompt -- hf dataset, including train HR images, train LR images downscale factor x8 and prompts. \
> datasets/DIV2K_train_dataset_x8_withoutprompt -- hf dataset, including train HR images, train LR images downscale factor x8. \
> datasets/DIV2K_train_dataset_x16_withprompt -- hf dataset, including train HR images, train LR images downscale factor x16 and prompts. \
> datasets/DIV2K_train_dataset_x16_withoutprompt -- hf dataset, including train HR images, train LR images downscale factor x16. \
>
> datasets/DIV2K_valid_HR -- 0801.png, 0802.png, ..., 0900.png, valid HR images \
> datasets/DIV2K_valid_LR_x4 -- valid LR images, downscale factor x4 \
> datasets/DIV2K_valid_LR_x8 -- valid LR images, downscale factor x8 \
> datasets/DIV2K_valid_LR_x16 -- valid LR images, downscale factor x16 \
> datasets/DIV2K_valid_prompt_short -- 0801.txt, 0802.txt, ..., 0900.txt, prompts for valid images \

## 🏰 Model Zoo

We provid the pre-trained LoRA checkpoints, download [here](https://box.nju.edu.cn/d/605bcdc252bb4a9b8462/)

## ⚡️ Quick Inference

You can inference on your own image with a pre-trained LoRA weight. Simply run:
```bash
./inference.sh  # Change the data paths as needed
```

## ⚙️ Training FLUX_SR
1. Log in to Hugging Face
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 Canny [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

2. Download DIV2K HR data
- [DIV2K Train data](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- [DIV2k Validation data](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)

3. Build FLUX_SR dataset

You can simply use our preprocessed dataset to train your FLUX_SR or build your own dataset.

First generate prompts, we use OpenAI GPT-4o to generate prompts and we provide a script to batch generate prompts. Run
```bash
python ./prompt_scr/batch_generate.py
```
You can use a more powerful model to generate prompts.

> **Token limit**
>
> Due to the token length limitation of the CLIP model, it is necessary to check the length of all prompts. If a prompt exceeds 77 tokens, it needs to be regenerated. \
> Run
```bash
python ./prompt_scr/test_tokens.py  
```
> to get a list of prompts that need to be shorten. \
> Then run
```bash
python ./prompt_scr/shorten.py
```
> to shorten the prompts. 

**You may need to repeat the above process several times to obtain a prompt that meets the length limitation.**

To build a hf dataset for training, run
```bash
./dataset_scr/preprocess.sh  # Change the data paths as needed
```

4. Fine-tuning FLUX with LoRA
```bash
accelerate launch --config_file ./configs/accelerate_config.yaml train_with_lora.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Canny-dev" \
    --local_dataset_name="./datasets/DIV2K_train_dataset_x4_withprompt" \
    --output_dir="./results/train_with_prompt_x4" \
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
    --offload \
    --seed="42" \
    --hub_token="" \
    --proportion_empty_prompts=0.0 \
    --tracker_project_name="train_with_prompt_x4"
```

    The training script exposes additional CLI args that might be useful to experiment with:

    * `use_lora_bias`: When set, additionally trains the biases of the `lora_B` layer. 
    * `train_norm_layers`: When set, additionally trains the normalization scales. Takes care of saving and loading.
    * `lora_layers`: Specify the layers you want to apply LoRA to. If you specify "all-linear", all the linear layers will be LoRA-attached.

5. Test on DIV2K validation set

```bash
./test.sh  # Change the args as needed
```


