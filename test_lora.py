import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchvision import transforms
from math_utils import *
from PIL import Image


pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda:1")
pipe.load_lora_weights(
    "/workspace/image_reso_old/lora_results/train_without_prompt_x16/checkpoint-600", adapter_name="resolution")
pipe.set_adapters("resolution", 0.85)

prompt = "A rockhopper penguin stands on rugged, dark gray rocks, slightly off-center to the left. Its yellow crest contrasts with its black and white body. The penguin's wings are slightly spread, and its pink feet are visible. The rocky background is textured with shades of gray and black, creating a natural habitat setting."
control_image = load_image(
    "/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_LR_x4/0801.png")
HR_image = Image.open(
    "/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_valid_HR/0801.png").convert("RGB")

# image_transforms = transforms.Compose(
#     [
#         transforms.Resize((1024, 1024),
#                           interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ]
# )

# control_image = control_image.convert("RGB")
# control_image = image_transforms(control_image)
control_image = control_image.convert("RGB")

HR_image = HR_image.resize((1024, 1024), Image.BILINEAR)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    # height=HR_image.size[1],
    # width=HR_image.size[0],
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output_804_x4_800_onlypinkprompt.png")


psnr_5, psnr, ssim_ = calculate_psnr_ssim(image, HR_image)
print(psnr_5, psnr, ssim_)
