from pipeline import CustomPipeline
from diffusers import LCMScheduler
import torch
from transformers import CLIPTextModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from torchvision import transforms
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.attention_processor import AttnProcessor2_0
import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image


adapter_id = "latent-consistency/lcm-lora-sdv1-5"
base_model = "lambdalabs/sd-image-variations-diffusers"

device = torch.device("mps")
generator = torch.Generator(device).manual_seed(12312)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="image_encoder").to(device=device)
feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")
do_classifier_free_guidance = True
dtype = next(image_encoder.parameters()).dtype


def load_model():
    pipe = CustomPipeline.from_pretrained(
        base_model,
        # "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        # try usign bits and bytes for faster inference using 8 bit precision
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device)

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.disable_xformers_memory_efficient_attention()
    print("loading lora weights")

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    return pipe

pipe = load_model().to(device)

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

impath = Path("./output_0.11.jpg").expanduser()
img = load_img(impath).unsqueeze(0).to("mps")

prompt = "A photo of the eiffel tower"
alternate_prompt = "a photo of the eiffel tower at night"
# text_embeddings = pipe.get_text_embedding(prompt)
# alternate_embeddings = pipe.get_text_embedding(alternate_prompt)

image_latents = pipe.get_image_latents(img, rng_generator=torch.Generator(device=pipe.device).manual_seed(0))

print(image_latents.shape)

reversed_latents = pipe.forward_diffusion(
    latents=image_latents,
    text_embeddings=None,
    guidance_scale=1,
    num_inference_steps=5,
)


# Reconstruct Latents

# def latents_to_imgs(latents):
#     x = pipe.decode_image(latents)
#     x = pipe.torch_to_numpy(x)
#     x = pipe.numpy_to_pil(x)
#     return x



# output = latents_to_imgs(image_latents)[0]
# output.save("output.jpg")