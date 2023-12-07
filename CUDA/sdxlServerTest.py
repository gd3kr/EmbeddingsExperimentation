from diffusers import DiffusionPipeline, LCMScheduler, AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

model_id =  "sd-dreambooth-library/herge-style"

# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("mps")


pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

prompt = "best quality, high quality"
image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
# images = pipe(
#     prompt=prompt,
    
#     num_inference_steps=1,
#     guidance_scale=1,
# ).images[0]

images = pipe(ip_adapter_image=image, prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]



images.save("generated_image.png")
