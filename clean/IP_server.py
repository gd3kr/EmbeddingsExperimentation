# if imports fail install diffusers from github (pip install git+https://github.com/huggingface/diffusers.git)

import io
import requests
import time
import base64
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.loaders import IPAdapterMixin
import inspect
from typing import Any, Callable, List, Optional, Union
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import load_image
from diffusers.models import ImageProjection
from torchvision import transforms
from diffusers import AutoencoderTiny
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers import *
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput
from flask import Flask, request, jsonify, Response
from PIL import Image
# BytesIO
from io import BytesIO
from flask_cors import CORS, cross_origin
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
 

    

def load_model():
    # model_id =  "sd-dreambooth-library/herge-style"
    model_id = "Lykon/DreamShaper"
    pipe = CustomPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device="cuda", dtype=torch.float16)
    pipe.safety_checker = None

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", torch_dtype=torch.float16)


    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", torch_dtype=torch.float16)
    
    # pipe.unet.set_attn_processor(AttnProcessor2_0())

    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    # pipe.disable_xformers_memory_efficient_attention()
    return pipe
    


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


test_image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
def generate_image(latents, prompt_embeds, negative_prompt_embeds): # takes in latents as input and generates an image with SD
  # ATTENTION: FOR SOME REASON, SETTING GUIDANCE SCALE TO 1 ABSOLUTELY FUCKS UP THE WHOLE PIPELINE, TREASURE YOUR BRAIN CELLS

    # Generate a blank black image of size 512x512
  blank_image = torch.zeros((1, 3, 512, 512), device="cuda")
    # Convert the blank black image to a PIL image
  blank_image_pil = transforms.ToPILImage()(blank_image.squeeze(0))
    
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
    
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
config.enable_cuda_graph = True

print("COMPILING WITH stable-fast...")
pipe = compile(pipe, config)

print("COMPILED!!!!!")

# exit(1)


# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
# image = load_image("https://pbs.twimg.com/media/F2_uILUXwAA0erl.jpg")
# image = load_image("https://pbs.twimg.com/media/GCiPBxfWgAAWByB?format=jpg&name=medium")
# image = load_image("https://pbs.twimg.com/media/GCc-Uw5WYAAvec2?format=jpg&name=large")
# image = load_image("https://upload.wikimedia.org/wikipedia/commons/0/02/Great_Wave_off_Kanagawa_-_reversed.png")
# image = load_image("https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/MA_00162721_yqcuno.jpg")
# image = load_image("https://images.wsj.net/im-398311?width=1280&size=1")
# image = load_image("https://www.byronmusic.com.au/cdn/shop/products/martinez-small-body-acoustic-guitar-spruce-top-mf-25-nst-28769932050627_1200x.jpg?v=1651056022")




image_embeds, negative_image_embeds = CustomPipeline.encode_image(
                pipe, image, device, 1, output_hidden_states=False
                )
    # concat
latent = torch.cat([negative_image_embeds, image_embeds])
latent = mix_images(latent, latent, 0.5)



hq_prompt_embeds, hq_negative_prompt_embeds = CustomPipeline.encode_prompt(
    pipe,
        "high quality, unreal engine, masterful composition",
        device,
        1,
        True,
)

lego_prompt_embeds, lego_negative_prompt_embeds = CustomPipeline.encode_prompt(
        pipe,
        "purple colored water, high quality, unreal engine, masterful composition",
        device,
        1,
        True,
)

prompt_latents = [hq_prompt_embeds, lego_prompt_embeds]
prompt_latents = process_latents(prompt_latents, "slerp")
for i, pl in enumerate(prompt_latents):
    image = generate_image(latent, pl*1.2, hq_negative_prompt_embeds).images[0]
    # save image
    image.save(f"test_images/{i}.jpg")



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

    # image = image.resize((512, 512))
    
    image_embeds, negative_image_embeds = CustomPipeline.encode_image(
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



prompt_embeds, negative_prompt_embeds = CustomPipeline.encode_prompt(
pipe,
    "screenshot from disney pixar, high quality, masterful composition",
    device,
    1,
    True,
)

@app.route('/avg', methods=['GET'])
def generate_avg_time():
    image = load_image("https://is1-ssl.mzstatic.com/image/thumb/Purple1/v4/a7/75/85/a77585b2-1818-46cc-0e18-2669cb1869a2/source/512x512bb.jpg")
    image = image.resize((512, 512))
    

    image_embeds, negative_image_embeds = CustomPipeline.encode_image(
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

images = {}


@app.route("/pregenerate", methods=['POST'])
def pregenerate():
    start_time = time.time()
    data = request.get_json()
    id_a = data['id_a']
    id_b = data['id_b']
    id_c = data['id_c']

    positions = data['positions'] # array of length 3, each element is a number between 0 and 1
    positions[0] = 0
    positions[2] = 1
    images = {}

    # Check if all latents are in the latents_store
    if id_a in latents_store and id_b in latents_store and id_c in latents_store:
        # Proceed with generation
        pass
    else:
        # Return an error response if any latent is missing
        return jsonify({'error': 'Missing latent representations in the store.'}), 400


    num_images = data['num_images'] # the number of images to generate tweening    

    # generates and saves images

    # mix between 0 and 1 for n images
    
    step_values = np.linspace(0, 1, num_images)
    images_key = id_a + id_b + id_c
    if images_key not in images:
        images[images_key] = []
    for step in step_values:
        # interpolated_latent = mix_images(latents_store[id_b], latents_store[id_c], step)

        # if step is between positions 0 and 1
        if step <= positions[0]:
            interpolated_latent = latents_store[id_a]
        elif step <= positions[1]:
            # interpolated_latent = mix_images(latents_store[id_a], latents_store[id_b], (step - positions[0]) / (positions[1] - positions[0]))
            interpolated_latent = slerp(latents_store[id_a], latents_store[id_b], (step - positions[0]) / (positions[1] - positions[0]))
        elif step <= positions[2]:
            # interpolated_latent = mix_images(latents_store[id_b], latents_store[id_c], (step - positions[1]) / (positions[2] - positions[1]))
            interpolated_latent = slerp(latents_store[id_b], latents_store[id_c], (step - positions[1]) / (positions[2] - positions[1]))
        else:
            interpolated_latent = latents_store[id_c]
            # inter

        # if step <= positions[1]:
        #     interpolated_latent = mix_images(latents_store[id_a], latents_store[id_b], step / positions[1])
        # elif step <= positions[2]:
        #     interpolated_latent = mix_images(latents_store[id_b], latents_store[id_c], (step - positions[1]) / (positions[2] - positions[1]))
        # else:
        #     interpolated_latent = latents_store[id_c]
        image = generate_image(interpolated_latent, prompt_embeds, negative_prompt_embeds).images[0]


        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_base64 = img_str.decode('utf-8')
        images[images_key].append(img_base64)

    end_time = time.time()

    print(f"Processing time: {end_time - start_time} `seconds")
    # Since Image objects are not JSON serializable, we need to convert them to a serializable format
    # Convert images to a list of base64 encoded strings
    

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
    # send as jpg image


    return Response(img_byte_arr, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()