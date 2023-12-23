import io
import requests

import matplotlib.pyplot as plt
import torch
from PIL import Image
from flask import Flask, request, jsonify, Response
from io import BytesIO
from torchvision import transforms
from transformers import (
    CLIPTextModel, 
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection, 
    AutoTokenizer, 
    CLIPTextModelWithProjection
)
from diffusers import LCMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models.attention_processor import AttnProcessor2_0
from pipeline import CustomPipeline


app = Flask(__name__)

adapter_id = "latent-consistency/lcm-lora-sdv1-5"
base_model = "lambdalabs/sd-image-variations-diffusers"


device = torch.device("mps")
generator = torch.Generator(device).manual_seed(0)

initial_seed = generator.initial_seed()
def reset_generator_seed():
    generator.manual_seed(initial_seed)

app.before_request(reset_generator_seed)




image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="image_encoder", torch_dtype=torch.float16).to(device=device)
feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor", torch_dtype=torch.float16)
do_classifier_free_guidance = True
dtype = next(image_encoder.parameters()).dtype


def encode_image(image_encoder, feature_extractor, dtype, image, text, device, num_images_per_prompt):
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
        torch_dtype=dtype,
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

    # text="flowers"
    output= encode_image(image_encoder, feature_extractor, dtype, tform(image).unsqueeze(0), "", device, 1)
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

def mix_images(image_a, image_b, mix_value):
    return image_a * (1 - mix_value) + image_b * mix_value

latents_store = {}
pipe = load_model().to(device)

@app.route('/store_latent', methods=['POST'])
def store_latent():
    data = request.get_json()
    image_url = data['image_url']
    image_id = data['id']
    
    if (not image_url or not image_id):
        return jsonify({'error': "bruh"}), 500

    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500

    latent = get_latents(image)
    latents_store[image_id] = latent

    return jsonify({'message': 'latent stored', 'id': image_id}), 200


@app.route('/slerp', methods=['POST'])
def slerp_route():

    data = request.get_json()
    id_a = data['id_a']
    id_b = data['id_b']
    n = float(data['mix_value'])

    if id_a not in latents_store or id_b not in latents_store:
        return jsonify({'error': 'one or both ids do not exist in the latent store'}), 400

    if not (0 <= n <= 1):
        return jsonify({'error': 'n must be between 0 and 1'}), 400

    start_embedding = latents_store[id_a]
    end_embedding = latents_store[id_b]
    interpolated_latent = slerp(start_embedding, end_embedding, n)
    interpolated_image = generate_image(interpolated_latent).images[0]

    img_byte_arr = io.BytesIO()
    interpolated_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(img_byte_arr, mimetype='image/jpeg')


@app.route('/mix', methods=['POST'])
def mix():
    data = request.get_json()
    id_a = data['id_a']
    id_b = data['id_b']
    mix_value = float(data['mix_value'])
    

    if not (0 <= mix_value <= 1):
        return jsonify({'error': 'mix_value must be between 0 and 1'}), 400

    if id_a not in latents_store or id_b not in latents_store:
        return jsonify({'error': 'one or both ids do not exist in the latent store'}), 400

    image_a_latent = latents_store[id_a]
    image_b_latent = latents_store[id_b]

    mixed_latent = mix_images(image_a_latent, image_b_latent, mix_value)
    mixed_image = generate_image(mixed_latent).images[0]

    img_byte_arr = io.BytesIO()
    mixed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    # send as jpg image


    return Response(img_byte_arr, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)