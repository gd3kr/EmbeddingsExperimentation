import io
import requests
import time
import numpy as np


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
import base64
from diffusers.utils import load_image
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
from flask_cors import CORS, cross_origin



import torch._dynamo
torch._dynamo.config.suppress_errors=True

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:5500"}})


adapter_id = "latent-consistency/lcm-lora-sdv1-5"
base_model = "lambdalabs/sd-image-variations-diffusers"


device = torch.device("cuda")
generator = torch.Generator(device).manual_seed(0)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="image_encoder").to(device=device)
feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")
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
        torch_dtype=torch.float16
        # try usign bits and bytes for faster inference using 8 bit precision
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device)
    pipe.safety_checker = None

    tiny = True
    lcm = True
    doCompile = False

    if tiny:
        from diffusers import AutoencoderTiny
        pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
        pipe.vae = pipe.vae.cuda()

    if lcm:
        # load and fuse lcm lora
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()

    if doCompile:
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
        #pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')
        pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
        pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

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
    output= encode_image(image_encoder, feature_extractor, dtype, tform(image).unsqueeze(0), "", device, 1).half()
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
  images = pipe(latents, num_inference_steps=4,  guidance_scale=1.2, num_images_per_prompt=1, generator=generator, height=400, width=400)
  return images

def mix_images(image_a, image_b, mix_value):
    return image_a * (1 - mix_value) + image_b * mix_value

latents_store = {}
pipe = load_model().to(device)

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers
    config.enable_xformers = True
    print("USING XFORMERS")
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
    print("USING TRITON")

except ImportError:
    print('Triton not installed, skip')
    
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
config.enable_cuda_graph = True

print("COMPILING WITH stable-fast...")
pipe = compile(pipe, config)


# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
image_1 = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
image_2 = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")

latent_1 = get_latents(image_1)
latent_2 = get_latents(image_2)
latent = mix_images(latent_1, latent_2, 0.5)



for _ in range(10):
    generate_image(latent).images[0]


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

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

@app.route('/avg', methods=['GET'])
def generate_avg_time():
    image_1 = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
    image_2 = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")

    latent_1 = get_latents(image_1)
    latent_2 = get_latents(image_2)
    start_time = time.time()
    for _ in range(20):
        latent = mix_images(latent_1, latent_2, 0.5)

        generate_image(latent).images[0]
    end_time = time.time()
    avg_time = (end_time - start_time) / 20
    print("Average time per image: ", avg_time)
    return jsonify({'average_time': avg_time}), 200


# images = {id(a+b): images[] }

images = {}

@app.route("/pregenerate", methods=['POST'])
def pregenerate():
    start_time = time.time()
    data = request.get_json()
    id_a = data['id_a']
    id_b = data['id_b']

    num_images = data['num_images'] # the number of images to generate tweening    

    # generates and saves images

    # mix between 0 and 1 for n images
    
    step_values = np.linspace(0, 1, num_images)
    for step in step_values:
        interpolated_latent = slerp(latents_store[id_a], latents_store[id_b], step)
        image = generate_image(interpolated_latent).images[0]
        images_key = id_a + id_b
        if images_key not in images:
            images[images_key] = []

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_base64 = img_str.decode('utf-8')
        images[images_key].append(img_base64)

    end_time = time.time()

    print(f"Processing time: {end_time - start_time} seconds")
    # Since Image objects are not JSON serializable, we need to convert them to a serializable format
    # Convert images to a list of base64 encoded strings
    

    return jsonify({'images': images[images_key]}), 200




@app.route('/mix', methods=['POST'])
def mix():
    start_time = time.time()
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


    end_time = time.time()

    print(f"Processing time: {end_time - start_time} seconds")
    return Response(img_byte_arr, mimetype='image/jpeg')



if __name__ == '__main__':
    app.run()