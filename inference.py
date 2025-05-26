# coding=utf-8

import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchvision import transforms
from math_utils import *
from PIL import Image
from glob import glob
import argparse


def eval(lora_weight_path, image_dir, prompt_dir, gt_dir, output_dir, step, device):
    os.makedirs(output_dir, exist_ok=True)
    lora_weight_path = os.path.join(lora_weight_path+'-'+str(step))

    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights(lora_weight_path, adapter_name="resolution")
    pipe.set_adapters("resolution", 0.85)

    base_name = os.path.splitext(os.path.basename(image_dir))[0]
    control_image = load_image(image_dir)
    HR_image = Image.open(gt_dir).convert("RGB")
    width, height = HR_image.size
    control_image = control_image.convert("RGB")

    with open(prompt_dir, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    image_save = pipe(
        prompt=prompt,
        control_image=control_image,
        height=height,
        width=width,
        num_inference_steps=50,
        guidance_scale=30.0,
        generator=torch.Generator().manual_seed(42),
    ).images[0]

    H, W = image_save.size

    HR_image_save = HR_image.resize((H, W), Image.BILINEAR)

    save_dir = os.path.join(output_dir, "eval_single_" +
                            base_name+"_"+str(step)+".png")
    image_save.save(save_dir)


if __name__ == '__main__':
    """
    evaluate a single LR images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_weight_dir', type=str,
                        default=None, help='trained lora weight path')
    parser.add_argument(
        '--image_dir', type=str, default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_LR_x4', help='Input image folder')
    parser.add_argument('--prompt_dir', type=str,
                        default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_prompt_short', help='Input prompt folder')
    parser.add_argument('--gt_dir', type=str,
                        default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_HR', help='Ground Truth HR image path')
    parser.add_argument('--output_dir', type=str,
                        default=None, help='Output estimated HR image save path')
    parser.add_argument('--step', type=int, default=100,
                        help='The step of checkpoint')
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    eval(args.lora_weight_dir, args.image_dir,
         args.prompt_dir, args.gt_dir, args.output_dir, args.step, args.device)
