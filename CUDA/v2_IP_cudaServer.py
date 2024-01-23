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
# torchvision transforms
from torchvision import transforms

from diffusers import AutoencoderTiny



import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers import *
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL
#import dict
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput
from flask import Flask, request, jsonify, Response
# PIL imgae
from PIL import Image
# BytesIO
from io import BytesIO

from flask_cors import CORS, cross_origin


from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile, CompilationConfig)
 



def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CustomPipeline(StableDiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            generator.manual_seed(0)  # Force the generator seed for reproducibility
            print("sampling random latents")
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            print("seed is: ", generator.initial_seed())
            print("shape of latents: " + str(shape))
        else:
            latents = latents.to(device)

        return latents


    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        


        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        
    
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)


        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        print("classifier free guidance is")
        print(self.do_classifier_free_guidance)


        # print("embedding prompt with settings:")
        # print("prompt: ", prompt)
        # print("device: ", device)
        # print("num_images_per_prompt: ", num_images_per_prompt )
        # print("do_classifier_free_guidance: ", self.do_classifier_free_guidance)
        # print("negative_prompt: ", negative_prompt)
        # print("prompt_embeds: ", prompt_embeds)
        # print("negative_prompt_embeds: ", negative_prompt_embeds)
        # print("lora scale: ", lora_scale)
        # print("clip_skip: ", self.clip_skip)
        # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     self.do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=lora_scale,
        #     clip_skip=self.clip_skip,
        # )

        # print("shape of prompt_embeds: " + str(prompt_embeds.shape))
        # print("shape of negative_prompt_embeds: " + str(negative_prompt_embeds.shape))


        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            print("embeddings prompt with prompt", prompt)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])


        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True

            print("output_hidden_state is: " + str(output_hidden_state))

            # # PRINT POSITIONAL ARGUMENTS
            # print("printing positional arguments")
            # print("device is: " + str(device))
            # print("num_images_per_prompt is: " + str(num_images_per_prompt))
            # print("output_hidden_state is: " + str(output_hidden_state))


            # image_embeds, negative_image_embeds = self.encode_image(
            #     ip_adapter_image, device, num_images_per_prompt, output_hidden_states=output_hidden_state
            # )
            # print("shape of the image embeds is: " + str(image_embeds.shape))
            # print("shape of the negative image embeds is: " + str(negative_image_embeds.shape))
            # if self.do_classifier_free_guidance:
            #     image_embeds = torch.cat([negative_image_embeds, image_embeds])
            #     print("shape of the image embeds after concatenation is: " + str(image_embeds.shape))
        
        print("output_hidden_state is: " + str(output_hidden_state))


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {}
        if ip_adapter_image is not None:
            added_cond_kwargs["image_embeds"] = input_image_embeds

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                #

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    

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
        prompt="made out of lego, colorful, rainbow",
        input_image_embeds=latents,
        ip_adapter_image=test_image,
        num_inference_steps=5,
        guidance_scale=1.2, #change to 1.9
        generator=generator,
        do_classifier_free_guidance=True, 
        # prompt_embeds=prompt_embeds,
        # negative_prompt_embeds=negative_prompt_embeds
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
    "high quality, unreal engine, masterful composition",
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