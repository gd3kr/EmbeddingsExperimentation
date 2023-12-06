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
from PIL import Image


adapter_id = "latent-consistency/lcm-lora-sdv1-5"
base_model = "lambdalabs/sd-image-variations-diffusers"


device = torch.device("cuda")
generator = torch.Generator(device).manual_seed(12312)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="image_encoder").to(device=device).half()
feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")
do_classifier_free_guidance = False
dtype = next(image_encoder.parameters()).dtype

print("dtype of the image encoder is: "+ str(dtype))


def encode_image(image_encoder, feature_extractor, dtype, do_classifier_free_guidance, image, device, num_images_per_prompt):
    if not isinstance(image, torch.Tensor):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values

    print("shape of the input image(tensor) is: "+ str(image.shape))
    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings

def load_model():
    pipe = CustomPipeline.from_pretrained(
        base_model,
        revision="v2.0",

        # try usign bits and bytes for faster inference using 8 bit precision
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device).half()


    pipe.unet.set_attn_processor(AttnProcessor2_0())


    pipe.enable_xformers_memory_efficient_attention()
    # pipe.disable_xformers_memory_efficient_attention()

    
    print("loading lora weights")

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    

    print("moving to cuda and compiling wiht oneflow")
    pipe = pipe.to("cuda")


    return pipe



def get_latents(image):

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

    output= encode_image(image_encoder, feature_extractor, dtype, do_classifier_free_guidance, tform(image).unsqueeze(0), device, 1)
    return output


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


def generate_image(latents): # takes in latents as input and generates an image with SD
  # ATTENTION: FOR SOME REASON, SETTING GUIDANCE SCALE TO 1 ABSOLUTELY FUCKS UP THE WHOLE PIPELINE, TREASURE YOUR BRAIN CELLS

  with torch.inference_mode():
    images = pipe(latents, num_inference_steps=4, guidance_scale=1.2, num_images_per_prompt=1, generator=generator)
    return images


def latent_processing():
    import glob
    import os
    oceans = []
    barren = []
    ocean_dir = './workspace/oceans'
    barren_dir = './workspace/barren'
    ocean_paths_jpg = glob.glob(os.path.join(ocean_dir, '*.jpg'))
    ocean_paths_png = glob.glob(os.path.join(ocean_dir, '*.png'))
    barren_paths_jpg = glob.glob(os.path.join(barren_dir, '*.jpg'))
    barren_paths_png = glob.glob(os.path.join(barren_dir, '*.png'))

    # Combine the lists of paths
    ocean_paths = ocean_paths_jpg + ocean_paths_png
    barren_paths = barren_paths_jpg + barren_paths_png

    ocean_latents = []
    for image_path in ocean_paths:
        image = Image.open(image_path).convert('RGB')
        ocean_latents.append(get_latents(image))

    barren_latents = []
    for image_path in barren_paths:
        image = Image.open(image_path).convert('RGB')
        barren_latents.append(get_latents(image))


    mean_ocean_embeddings = torch.mean(torch.stack(ocean_latents), dim=0)
    print(mean_ocean_embeddings.shape)


    mean_barren_embeddings = torch.mean(torch.stack(barren_latents), dim=0)
    print(mean_barren_embeddings.shape)
    return mean_ocean_embeddings, mean_barren_embeddings



# mean_ocean_embeddings, mean_barren_embeddings = latent_processing()

# # store the mean embeddings in a file
# torch.save(mean_ocean_embeddings, "mean_ocean_embeddings.pt")
# torch.save(mean_barren_embeddings, "mean_barren_embeddings.pt")

# exit(1)

# load the mean embeddings from a file
mean_ocean_embeddings = torch.load("mean_ocean_embeddings.pt")
mean_barren_embeddings = torch.load("mean_barren_embeddings.pt")

pipe = load_model()
#Eiffel Tower
url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/1:1/w_3633,h_3633,c_limit/GettyImages-468366251.jpg"

#new york
# url = "https://www.travelandleisure.com/thmb/91pb8LbDAUwUN_11wATYjx5oF8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/new-york-city-evening-NYCTG0221-52492d6ccab44f328a1c89f41ac02aea.jpg"

#collosseum
# url = "https://miro.medium.com/v2/resize:fit:1198/1*W_quBAVAMu8a4n1Sg07Xvw.jpeg"

# Taj Mahal URL
# url = "https://th-thumbnailer.cdn-si-edu.com/NaExfGA1op64-UvPUjYE5ZqCefk=/fit-in/1600x0/filters:focal(1471x1061:1472x1062)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/b6/30/b630b48b-7344-4661-9264-186b70531bdc/istock-478831658.jpg"

#fish
# url = "https://media.istockphoto.com/id/699306668/photo/underwater-scene-with-coral-reef-and-tropical-fish.jpg?s=612x612&w=0&k=20&c=HstAcsVjlUp1vNLni0_YpYRRODiBrXMC2FGUKKKZZuk="

image = Image.open(requests.get(url, stream=True).raw)
start_embedding = get_latents(image)
positive_to_negative = + mean_ocean_embeddings - mean_barren_embeddings

images = [] # List to store generated images
generation_time = [] # List to store generation times

import time

for t in torch.linspace(0, 1, 10):
    embedding = slerp(start_embedding, start_embedding + positive_to_negative, t)
    print(f'negative × {t:.2f}')

    # measure time
    start = time.time()
    img = generate_image(embedding).images[0]
    end = time.time()
    # save image
    img.save(f"output_{t:.2f}.jpg")
    images.append(img)
    generation_time.append(end - start)
# Concatenate images horizontally

print("average generation time: ", sum(generation_time)/len(generation_time))
concatenated_image = Image.new('RGB', (images[0].width * len(images), images[0].height))
x_offset = 0
for img in images:
    concatenated_image.paste(img, (x_offset, 0))
    x_offset += img.width

# Display the concatenated image
plt.figure(figsize=(50, 5))  # Adjust the figure size as needed
# save image
concatenated_image.save("output.jpg")

plt.imshow(concatenated_image)
plt.axis('off')  # Hide axes
plt.show()

