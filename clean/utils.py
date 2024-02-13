import time
import torch
from diffusers import *
from diffusers.utils import load_image
from pipeline import LatentSurfingPipeline

def generate_image(pipe, latents, prompt_embeds, negative_prompt_embeds, generator):
  test_image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
  start = time.time()
  images = pipe(
        # prompt="made out of lego, colorful, rainbow",
        input_image_embeds=latents,
        ip_adapter_image=test_image,
        num_inference_steps=5,
        guidance_scale=1.2, #change to 1.9
        generator=generator,
        do_classifier_free_guidance=True, 
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds
        # height=512,
        # width=512,
    )
  end = time.time()
  print("PIPE TIME: ", end-start)
  return images

def mix_images(image_a, image_b, mix_value):
    return image_a * (1 - mix_value) + image_b * mix_value


def slerp(a, b, n, eps=1e-8):
    a_norm = a / torch.norm(a)
    b_norm = b / torch.norm(b)
    omega = torch.acos((a_norm * b_norm).sum()) + eps
    so = torch.sin(omega)
    return (torch.sin((1.0 - n) * omega) / so) * a + (torch.sin(n * omega) / so) * b

def process_latents(latents, operation): # apply mean, average, etc
    if operation == "mean":
      return torch.stack(latents).mean(dim=0)
    if operation == "sum":
      return torch.sum(torch.stack(latents), dim=0)
    if operation == "slerp":
      start_embedding = latents[0]
      end_embedding = latents[1]
      interpolated = []
      for t in torch.linspace(0, 1, 10):
        latent = slerp(start_embedding, end_embedding, t)
        interpolated.append(latent)
      return torch.stack(interpolated)
    if operation == "mix":
        start_embedding = latents[0]
        end_embedding = latents[1]
        interpolated = []
        for t in torch.linspace(0, 1, 10):
            latent = mix_images(start_embedding, end_embedding, t)
            interpolated.append(latent)
