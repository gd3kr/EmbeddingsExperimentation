# if imports fail install diffusers from github (pip install git+https://github.com/huggingface/diffusers.git)

import io
import requests
import time
import base64
import numpy as np
from diffusers.utils import load_image
from torchvision import transforms
from diffusers import AutoencoderTiny
import torch
from diffusers import *
from flask import Flask, request, jsonify, Response
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)

from pipeline import LatentSurfingPipeline
from utils import generate_image, mix_images, slerp, process_latents


def load_model():
    # model_id =  "sd-dreambooth-library/herge-style"
    model_id = "Lykon/DreamShaper"
    pipe = LatentSurfingPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device="cuda", dtype=torch.float16)
    pipe.safety_checker = None
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", torch_dtype=torch.float16)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", torch_dtype=torch.float16)
    
    # pipe.unet.set_attn_processor(AttnProcessor2_0())
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    return pipe


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

latents_store = {}
device = "cuda"
generator = torch.Generator(device=device).manual_seed(42)

pipe = load_model().to("cuda")

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers
    config.enable_xformers = False
    print("USING XFORMERS")
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
    print("USING TRITON")

except ImportError:
    print('Triton not installed, skip')
    
config.enable_cuda_graph = True

print("compiling with stable-fast...")
pipe = compile(pipe, config)

print("COMPILED!!!!!")


# load an image
image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
# image = load_image("https://pbs.twimg.com/media/F2_uILUXwAA0erl.jpg")
# image = load_image("https://pbs.twimg.com/media/GCiPBxfWgAAWByB?format=jpg&name=medium")
# image = load_image("https://pbs.twimg.com/media/GCc-Uw5WYAAvec2?format=jpg&name=large")
# image = load_image("https://upload.wikimedia.org/wikipedia/commons/0/02/Great_Wave_off_Kanagawa_-_reversed.png")
# image = load_image("https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/MA_00162721_yqcuno.jpg")


images = {}

# Text prompt embedding
prompt_embeds, negative_prompt_embeds = LatentSurfingPipeline.encode_prompt(
pipe,
    "screenshot from disney pixar, high quality, masterful composition",
    device,
    1,
    True,
)



@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

@app.route('/store_latent', methods=['POST'])
def store_latent():
    print("Request received")
    data = request.get_json()
    image_url = data.get('image_url', None)
    image_id = data.get('id', None)
    image_b64 = data.get('image_b64', None)

    print("Storing latent with id: ", image_id)
    print("image_url: ", image_url)
    # print("image_b64: ", image_b64[:10])
    print("image_id: ", image_id)

    if (not (image_url or image_b64) or not image_id):
        return jsonify({'error': 'image_url, image_b64, and id are required'}), 400

    try:
        if image_b64:
            try:
                image_data = base64.b64decode(image_b64)
            
                image = Image.open(BytesIO(image_data)).convert('RGB')
            except IOError as e:
                return jsonify({'error': 'Cannot identify image file'}), 400
        else:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500
    
    image_embeds, negative_image_embeds = LatentSurfingPipeline.encode_image(
                pipe, image, device, 1, output_hidden_states=False
                )
    # concat
    image_latents = torch.cat([negative_image_embeds, image_embeds])

    latents_store[image_id] = image_latents

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


@app.route('/avg', methods=['GET'])
def generate_avg_time():
    image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
    image = image.resize((512, 512))
    

    image_embeds, negative_image_embeds = LatentSurfingPipeline.encode_image(
                pipe, image, device, 1, output_hidden_states=False
                )
    # concat
    latent = torch.cat([negative_image_embeds, image_embeds])
    start_time = time.time()
    for _ in range(5):
        generate_image(latent, prompt_embeds, negative_prompt_embeds).images[0]
    end_time = time.time()
    avg_time = (end_time - start_time) / 5
    return jsonify({'average_time': avg_time}), 200


@app.route("/pregenerate", methods=['POST'])
def pregenerate():
    start_time = time.time()
    data = request.get_json()
    id_a = data['id_a']
    id_b = data['id_b']
    id_c = data['id_c']
    num_images = data['num_images']
    

    positions = data['positions'] # array of length 3, each element is a number between 0 and 1
    positions[0] = 0
    positions[2] = 1
    images = {}

    # Check if all latents are in the latents_store
    if id_a in latents_store and id_b in latents_store and id_c in latents_store:
        # Proceed with generation
        pass
    else:
        # Return an error response if something is missing
        return jsonify({'error': 'Missing latent representations in the store.'}), 400
    
    step_values = np.linspace(0, 1, num_images)
    images_key = id_a + id_b + id_c
    if images_key not in images:
        images[images_key] = []
    for step in step_values:

        # if step is between positions 0 and 1
        if step <= positions[0]:
            interpolated_latent = latents_store[id_a]
        elif step <= positions[1]:
            interpolated_latent = slerp(latents_store[id_a], latents_store[id_b], (step - positions[0]) / (positions[1] - positions[0]))
        elif step <= positions[2]:
            interpolated_latent = slerp(latents_store[id_b], latents_store[id_c], (step - positions[1]) / (positions[2] - positions[1]))
        else:
            interpolated_latent = latents_store[id_c]

        image = generate_image(interpolated_latent, prompt_embeds, negative_prompt_embeds).images[0]


        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_base64 = img_str.decode('utf-8')
        images[images_key].append(img_base64)

    end_time = time.time()

    print(f"Processing time: {end_time - start_time} `seconds")
    # Since Image objects are not JSON serializable, we need to convert them to a serializable format

    return jsonify({'images': images[images_key]}), 200

@app.route('/mix', methods=['POST'])
def mix():
    print("request received")
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

    print("latents acquired, generating Images...")
    mixed_image = generate_image(mixed_latent).images[0]

    img_byte_arr = io.BytesIO()
    mixed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(img_byte_arr, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()