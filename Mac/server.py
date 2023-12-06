from pipeline import CustomPipeline
from diffusers import LCMScheduler
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
generator = torch.Generator(device).manual_seed(42)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="image_encoder").to(device=device)
feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")
do_classifier_free_guidance = True
dtype = next(image_encoder.parameters()).dtype


def encode_text(text):
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device=device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    inputs = tokenizer([text], padding=True, return_tensors="pt").to(device)

    outputs = model(**inputs)
    text_embeds = outputs.text_embeds

    return text_embeds

def encode_image(image_encoder, feature_extractor, dtype, image, text, device, num_images_per_prompt):
    if not isinstance(image, torch.Tensor):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values

    text_embeddings = encode_text(text)
    print("shape of the input image(tensor) is: "+ str(image.shape))
    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)


    if do_classifier_free_guidance:
        print("doing classifier free guidance")
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings

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

    text="flowers"
    output= encode_image(image_encoder, feature_extractor, dtype, tform(image).unsqueeze(0), text, device, 1)
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

# store the mean embeddings in a file
# torch.save(mean_ocean_embeddings, "mean_ocean_embeddings.pt")
# torch.save(mean_barren_embeddings, "mean_barren_embeddings.pt")

# exit(1)

# load the mean embeddings from a file
mean_ocean_embeddings = torch.load("mean_ocean_embeddings.pt")
mean_barren_embeddings = torch.load("mean_barren_embeddings.pt")

pipe = load_model().to(device)
#Eiffel Tower
# url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/1:1/w_3633,h_3633,c_limit/GettyImages-468366251.jpg"

#new york
# url = "https://www.travelandleisure.com/thmb/91pb8LbDAUwUN_11wATYjx5oF8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/new-york-city-evening-NYCTG0221-52492d6ccab44f328a1c89f41ac02aea.jpg"

#collosseum
# url = "https://miro.medium.com/v2/resize:fit:1198/1*W_quBAVAMu8a4n1Sg07Xvw.jpeg"

# Taj Mahal URL
# url = "https://th-thumbnailer.cdn-si-edu.com/NaExfGA1op64-UvPUjYE5ZqCefk=/fit-in/1600x0/filters:focal(1471x1061:1472x1062)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/b6/30/b630b48b-7344-4661-9264-186b70531bdc/istock-478831658.jpg"

#fish
# url = "https://media.istockphoto.com/id/699306668/photo/underwater-scene-with-coral-reef-and-tropical-fish.jpg?s=612x612&w=0&k=20&c=HstAcsVjlUp1vNLni0_YpYRRODiBrXMC2FGUKKKZZuk="

#pinkney
url = "https://www.justinpinkney.com/img/YOBg9JfLdN-512.webp"

image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("ghibli.jpg").convert('RGB')

start_embedding = get_latents(image)
positive_to_negative = + mean_ocean_embeddings - mean_barren_embeddings
text_embedding_1 = encode_text("oil painting").unsqueeze(0)
text_embedding_2 = encode_text("dslr leica photo").unsqueeze(0)



images = [] # List to store generated images
for t in torch.linspace(0,0.5, 10):   
    
    # embedding = start_embedding + (text_embedding_1 - text_embedding_2) * t

    # weighted mean
    embedding = start_embedding*(1-t) + text_embedding_2 * t

    # embedding = slerp(start_embedding, start_embedding + text_embeddings, t)
    print(f'negative × {t:.2f}')
    img = generate_image(embedding).images[0]
    # save image
    img.save(f"2_{t:.2f}.jpg")
    images.append(img)


# for i in range(10):
#    # merge embedding and direction without slerp
#     embedding = start_embedding + positive_to_negative * i / 10
#     print(f'negative × {i:.2f}')
#     img = generate_image(embedding).images[0]
#     # save image
#     img.save(f"output_{i:.2f}.jpg")
#     images.append(img)

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

