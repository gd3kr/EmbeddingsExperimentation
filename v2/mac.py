from pipeline import V2Variation
from diffusers import LCMScheduler, StableDiffusionImageVariationPipeline
import torch
from transformers import CLIPTextModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from torchvision import transforms
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection
from diffusers.models.attention_processor import AttnProcessor2_0
import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
from PIL import Image


adapter_id = "latent-consistency/lcm-lora-sdv1-5"
base_model = "lambdalabs/sd-image-variations-diffusers"


device = torch.device("mps")

device = "mps"
pipe = V2Variation.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device)
pipe = pipe.to(device)

print("loading lora weights")

pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

im = Image.open("./ghibli.jpg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)

out = pipe(inp, guidance_scale=3)
out["images"][0].save("output.jpg")

