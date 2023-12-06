import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModel
from pipeline import CustomPipeline
from diffusers import LCMScheduler
import glob
import os

# Initialization
device = torch.device("mps")
generator = torch.Generator(device).manual_seed(25436)

# Load the model components
def load_model_components():
    pipe = CustomPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="v2.0")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="image_encoder"
    ).to(device=device)
    feature_extractor = CLIPImageProcessor.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="feature_extractor"
    )
    
    return pipe, image_encoder, feature_extractor

pipe, image_encoder, feature_extractor = load_model_components()

def encode_image(image, **kwargs):
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)
    image_encoder = kwargs.get("image_encoder", None)
    feature_extractor = kwargs.get("feature_extractor", None)

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values

    print("shape of the input image(tensor) is: "+ str(image.shape))
    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    # # duplicate image embeddings for each generation per prompt, using mps friendly method
    # bs_embed, seq_len, _ = image_embeddings.shape
    # image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    # image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings

def load_model():
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = CustomPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.image_processor = CLIPVisionModelWithProjection.from_config(pipe.image_processor.config)
    pipe.feature_extractor = CLIPImageProcessor.from_config(pipe.feature_extractor.config)
    pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.disable_xformers_memory_efficient_attention()
    print("loading lora weights")

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

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

    encode_args = {
        "dtype": next(image_encoder.parameters()).dtype,
        "device": device,
        "num_images_per_prompt": 1, 
        "do_classifier_free_guidance": True,
        "image_encoder": image_encoder,
        "feature_extractor": feature_extractor,
    }

    output= encode_image(tform(image).unsqueeze(0), **encode_args)
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
  images = pipe(latents, num_inference_steps=4, guidance_scale=1.2, num_images_per_prompt=1, generator=generator)
  return images

# Function to process and store latents
def latent_processing():
    oceans = []
    barren = []
    ocean_dir = './workspace/oceans'
    barren_dir = './workspace/barren'
    ocean_paths = glob.glob(os.path.join(ocean_dir, '*.[jp][np]g'))
    barren_paths = glob.glob(os.path.join(barren_dir, '*.[jp][np]g'))

    ocean_latents = [get_latents(Image.open(path).convert('RGB')) for path in ocean_paths]
    barren_latents = [get_latents(Image.open(path).convert('RGB')) for path in barren_paths]

    mean_ocean_embeddings = torch.mean(torch.stack(ocean_latents), dim=0)
    mean_barren_embeddings = torch.mean(torch.stack(barren_latents), dim=0)

    return mean_ocean_embeddings, mean_barren_embeddings

# mean_ocean_embeddings, mean_barren_embeddings = latent_processing()

# # Save the mean embeddings
# torch.save(mean_ocean_embeddings, "mean_ocean_embeddings.pt")
# torch.save(mean_barren_embeddings, "mean_barren_embeddings.pt")

# exit(1)

mean_ocean_embeddings = torch.load("mean_ocean_embeddings.pt")
mean_barren_embeddings = torch.load("mean_barren_embeddings.pt")

print(pipe)


import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch

#Eiffel Tower
url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/1:1/w_3633,h_3633,c_limit/GettyImages-468366251.jpg"

#new york
# url = "https://www.travelandleisure.com/thmb/91pb8LbDAUwUN_11wATYjx5oF8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/new-york-city-evening-NYCTG0221-52492d6ccab44f328a1c89f41ac02aea.jpg"

#collosseum
# url = "https://miro.medium.com/v2/resize:fit:1198/1*W_quBAVAMu8a4n1Sg07Xvw.jpeg"

# Taj Mahal URL
# url = "https://th-thumbnailer.cdn-si-edu.com/NaExfGA1op64-UvPUjYE5ZqCefk=/fit-in/1600x0/filters:focal(1471x1061:1472x1062)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/b6/30/b630b48b-7344-4661-9264-186b70531bdc/istock-478831658.jpg"

image = Image.open(requests.get(url, stream=True).raw)
start_embedding = get_latents(image)
positive_to_negative = mean_ocean_embeddings - mean_barren_embeddings

images = [] # List to store generated images
for t in torch.linspace(0, 1, 10):
    embedding = slerp(start_embedding, start_embedding + positive_to_negative, t)
    print(f'negative × {t:.2f}')
    img = generate_image(embedding).images[0]
    # save image
    img.save(f"output_{t:.2f}.jpg")
    images.append(img)
# Concatenate images horizontally
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


