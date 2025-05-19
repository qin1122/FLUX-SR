import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchvision import transforms
from math_utils import *
from PIL import Image
from glob import glob
import argparse


def load_validation_set(image_dir, prompt_dir, gt_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*')))
    validation_images = []
    validation_prompts = []
    gt_images = []

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        prompt_path = os.path.join(prompt_dir, base_name + ".txt")
        gt_path = os.path.join(gt_dir, base_name + ".png")

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(
                f"Prompt file not found for image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                f"GT file not found for image: {image_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        validation_images.append(image_path)
        validation_prompts.append(prompt)
        gt_images.append(gt_path)

    return validation_images, validation_prompts, gt_images


def test(lora_weight_path, image_dir, prompt_dir, gt_dir, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    lora_weight_path = os.path.join(lora_weight_path+'-'+str(step))

    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda:3")
    pipe.load_lora_weights(lora_weight_path, adapter_name="resolution")
    pipe.set_adapters("resolution", 0.85)

    validation_images, validation_prompts, gt_images = load_validation_set(
        image_dir, prompt_dir, gt_dir)

    accumulate_psnr = 0.0
    accumulate_psnr_5 = 0.0
    accumulate_ssim = 0.0
    total_num = 0.0

    for validation_image, validation_prompt, gt_image in zip(validation_images, validation_prompts, gt_images):
        base_name = os.path.splitext(os.path.basename(validation_image))[0]
        control_image = load_image(validation_image)
        HR_image = Image.open(gt_image).convert("RGB")
        width, height = HR_image.size
        control_image = control_image.convert("RGB")

        image_save = pipe(
            prompt=validation_prompt,
            control_image=control_image,
            height=height,
            width=width,
            num_inference_steps=50,
            guidance_scale=30.0,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        H, W = image_save.size
        HR_image = HR_image.resize((H, W), Image.BILINEAR)

        save_dir = os.path.join(output_dir+"_"+str(step), "output_" +
                                base_name+".png")
        image_save.save(save_dir)

        psnr_5, psnr, ssim_ = calculate_psnr_ssim(image_save, HR_image)
        accumulate_psnr += psnr
        accumulate_psnr_5 += psnr_5
        accumulate_ssim += ssim_
        total_num += 1

    print(f"average psnr: {accumulate_psnr/total_num}, average psnr_5: {accumulate_psnr_5/total_num}, average ssim: {accumulate_ssim/total_num}")


if __name__ == '__main__':
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
    args = parser.parse_args()

    test(args.lora_weight_dir, args.image_dir,
         args.prompt_dir, args.gt_dir, args.output_dir, args.step)
