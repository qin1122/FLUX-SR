import torch
from controlnet_aux import CannyDetector, HEDdetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
import warnings
warnings.simplefilter("ignore", FutureWarning)

# pipe = FluxControlPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "A black-and-white penguin with yellow-orange head streaks and a pink beak stands slightly off-center on rugged dark gray rocks, wings slightly extended, surrounded by textured stone formations."
control_image = load_image(
    "/root/image_reso/dataset/DIV2K_valid_LR_x32/0803.png")

processor = CannyDetector()
processor1 = HEDdetector.from_pretrained("lllyasviel/Annotators")
control_image = processor(control_image, low_threshold=50,
                          high_threshold=200, detect_resolution=40, image_resolution=1024)
control_image = processor1(control_image)
control_image.save("hed_image_803.png")
# image = pipe(
#     prompt=prompt,
#     control_image=control_image,
#     height=1024,
#     width=1024,
#     num_inference_steps=50,
#     guidance_scale=30.0,
# ).images[0]
# image.save("output.png")
