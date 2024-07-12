#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import argparse
import functools
import gc
import logging
import math
import os
import os.path as osp
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import safetensors

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
#from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
#from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.models import ImageProjection
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_tryon
from cp_dataset import CPDatasetV2 as CPDataset, UserDataDataset
from parser_args import parse_args
from model_container import ModelContainer

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from utils import combine_images_horizontally, combine_images_vertically, is_in_range

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)
# if is_torch_npu_available():
#     torch.npu.config.allow_internal_format = False

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


# def save_model_card(
#     repo_id: str,
#     images: list = None,
#     validation_prompt: str = None,
#     base_model: str = None,
#     dataset_name: str = None,
#     repo_folder: str = None,
#     vae_path: str = None,
# ):
#     img_str = ""
#     if images is not None:
#         for i, image in enumerate(images):
#             image.save(os.path.join(repo_folder, f"image_{i}.png"))
#             img_str += f"![img_{i}](./image_{i}.png)\n"

#     model_description = f"""
# # Text-to-image finetuning - {repo_id}

# This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
# {img_str}

# Special VAE used for training: {vae_path}.
# """

#     model_card = load_or_create_model_card(
#         repo_id_or_path=repo_id,
#         from_training=True,
#         license="creativeml-openrail-m",
#         base_model=base_model,
#         model_description=model_description,
#         inference=True,
#     )

#     tags = [
#         "stable-diffusion-xl",
#         "stable-diffusion-xl-diffusers",
#         "text-to-image",
#         "diffusers-training",
#         "diffusers",
#     ]
#     model_card = populate_model_card(model_card, tags=tags)

#     model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
   
def check_model_components(model):
    required_components = [
        'vae', 'text_encoder_one', 'text_encoder_two',
        'tokenizer_one', 'tokenizer_two', 'noise_scheduler',
        'image_encoder', 'ref_unet'
    ]
    
    missing_components = []
    for component in required_components:
        if not hasattr(model, component):
            missing_components.append(component)
    
    if missing_components:
        raise ValueError(f"Missing required model components: {', '.join(missing_components)}")
    

def check_sample_elements(sample):
    required_keys = [
        'cloth', 'caption', 'caption_cloth', 'pose_img', 
        'cloth_pure', 'inpaint_mask', 'image'
    ]
    
    missing_keys = [key for key in required_keys if key not in sample]
    
    if missing_keys:
        raise ValueError(f"Missing keys in sample: {missing_keys}")

def log_validation(unet, model, args, accelerator, weight_dtype, log_name, validation_dataloader):
    
    unet = accelerator.unwrap_model(unet)
    
    check_model_components(model)
    pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=model.vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = model.text_encoder_one,
            text_encoder_2 = model.text_encoder_two,
            tokenizer = model.tokenizer_one,
            tokenizer_2 = model.tokenizer_two,
            scheduler = model.noise_scheduler,
            image_encoder=model.image_encoder,
            torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.unet_encoder = model.ref_unet
    
        # Extract the images
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            image_logs = []
            for sample in validation_dataloader:
                check_sample_elements(sample)
                img_emb_list = []
                for i in range(sample['cloth'].shape[0]):
                    img_emb_list.append(sample['cloth'][i])
                
                prompt = sample["caption"]

                num_prompts = sample['cloth'].shape[0]                                        
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts

                image_embeds = torch.cat(img_emb_list,dim=0)

                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                
                
                    prompt = sample["caption_cloth"]
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    if not isinstance(prompt, List):
                        prompt = [prompt] * num_prompts
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * num_prompts


                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    
                    
                    inference_guidance_scale = [0.99, 2, 5]
                    output_images = []
                    
                    target_size = sample['image'].shape[2:]
                    sample['pose_img'] = F.interpolate(sample['pose_img'], size=target_size, mode='bilinear', align_corners=False)
                    sample['cloth_pure'] = F.interpolate(sample['cloth_pure'], size=target_size, mode='bilinear', align_corners=False)


                    for scale in inference_guidance_scale:
                        generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                        (images, inference_sampling_images, before_inference_images) = pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=args.inference_steps,
                            generator=generator,
                            strength=0.8,
                            pose_img=sample['pose_img'],
                            text_embeds_cloth=prompt_embeds_c,
                            cloth=sample['cloth_pure'].to(accelerator.device),
                            mask_image=sample['inpaint_mask'],
                            image=(sample['image'] + 1.0) / 2.0,
                            height=args.height,
                            width=args.width,
                            guidance_scale=scale,
                            ip_adapter_image=image_embeds,
                            inference_sampling_step=args.inference_sampling_step,
                        )
                        
                        # print(f"image: {images}, internal_images: {inference_sampling_images}")
                        # os._exit(os.EX_OK)
                        image = images[0]
                        internal_images = []
                        for before_var in before_inference_images:
                            internal_images.append(before_var[0])
                        for internal_var in inference_sampling_images:
                            internal_images.append(internal_var[0])
                        internal_images.append(image)
                        horizontal_image = combine_images_horizontally(internal_images)
                        output_images.append(horizontal_image)

                    # Combine the images into one
                    combined_image = combine_images_vertically(output_images)

                    image_logs.append({
                        "garment": sample["cloth_pure"], 
                        "model": sample['image'], 
                        "orig_img": sample['image'], 
                        "samples": combined_image, 
                        "prompt": prompt,
                        "inpaint mask": sample['inpaint_mask'],
                        "pose_img": sample['pose_img'],
                        })
                        
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    if not is_wandb_available():
                        raise ImportError("Make sure to install wandb if you want to use it for logging during validation.")
                    import wandb
                    formatted_images = []
                    for log in image_logs:
                        # logger.info("Adding image to tacker")
                        formatted_images.append(wandb.Image(log["garment"], caption="garment images"))
                        # formatted_images.append(wandb.Image(log["model"], caption="masked model images"))
                        formatted_images.append(wandb.Image(log["orig_img"], caption="original images"))
                        formatted_images.append(wandb.Image(log["inpaint mask"], caption="inpaint mask"))
                        formatted_images.append(wandb.Image(log["pose_img"], caption="pose_img"))
                        formatted_images.append(wandb.Image(log["samples"], caption=log["prompt"]))
                    tracker.log({log_name: formatted_images})
                else:
                    logger.warn(f"image logging not implemented for {tracker.name}")            

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def encode_image(model, image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(model.image_encoder.parameters()).dtype
    if not isinstance(image, torch.Tensor):
        image = model.feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    # print(f"encode image (initial): {image.dtype}")
    if output_hidden_states:
        image_enc_hidden_states = model.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        # print(f"encode image (after encoding): {image_enc_hidden_states.dtype}")
        
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        # print(f"encode image (after repeat_interleave): {image_enc_hidden_states.dtype}")
        
        uncond_image_enc_hidden_states = model.image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        # print(f"encode image (uncond encoding): {uncond_image_enc_hidden_states.dtype}")
        
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        # print(f"encode image (uncond after repeat_interleave): {uncond_image_enc_hidden_states.dtype}")

        # print(f"encode image (final): {image_enc_hidden_states.dtype}")


        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = model.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds


def compute_vae_encodings(pixel_values, vae):
    
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights

def load_model_with_zeroed_mismatched_keys(unet, pretrained_weights_path):
    # Load the pretrained weights
    # new weight as an initialized state
    # Determine the file type and load the pretrained weights
    if pretrained_weights_path.endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(pretrained_weights_path)
    elif pretrained_weights_path.endswith('.bin'):
        state_dict = torch.load(pretrained_weights_path)
    else:
        raise ValueError("Unsupported file type. Only .safetensors and .bin are supported.")
    
    # Initialize a new state dict for the model
    # The new unet data structure
    new_state_dict = unet.state_dict()
    
    # Iterate through the pretrained weights
    for key, value in state_dict.items():
        if key in new_state_dict and new_state_dict[key].shape == value.shape:
            new_state_dict[key] = value
        else:
            if key in unet.attn_processors.keys():
                # Initialize the ip adaptor for the model check https://github.com/tencent-ailab/IP-Adapter/blob/cfcdf8ce36f31e3d358b3c4c4b1bb78eab2854bd/tutorial_train_plus.py#L335
                layer_name = key.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet[layer_name + ".to_v.weight"],
                }
                new_state_dict[key] = weights
            else:
                print(f"Key {key} mismatched or not found in model. Initializing with zeros.")
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
    
    # Load the new state dict into the model
    unet.load_state_dict(new_state_dict)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
#     dtype = image.dtype
#     if self.vae.config.force_upcast:
#         image = image.float()
#         self.vae.to(dtype=torch.float32)

#     if isinstance(generator, list):
#         image_latents = [
#             retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
#             for i in range(image.shape[0])
#         ]
#         image_latents = torch.cat(image_latents, dim=0)
#     else:
#         image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

#     if self.vae.config.force_upcast:
#         self.vae.to(dtype)

#     image_latents = image_latents.to(dtype)
#     image_latents = self.vae.config.scaling_factor * image_latents

#     return image_latents



# def prepare_latents(
#     model,
#     batch_size,
#     num_channels_latents,
#     height,
#     width,
#     dtype,
#     device,
#     generator,
#     latents=None,
#     image=None,
#     timestep=None,
#     is_strength_max=True,
#     add_noise=True,
#     return_noise=False,
#     return_image_latents=False,
# ):
#     shape = (batch_size, num_channels_latents, height // model.vae_scale_factor, width // model.vae_scale_factor)
#     if isinstance(generator, list) and len(generator) != batch_size:
#         raise ValueError(
#             f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
#             f" size of {batch_size}. Make sure the batch size matches the length of the generators."
#         )

#     if (image is None or timestep is None) and not is_strength_max:
#         raise ValueError(
#             "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
#             "However, either the image or the noise timestep has not been provided."
#         )

#     if image.shape[1] == 4:
#         image_latents = image.to(device=device, dtype=dtype)
#         image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
#     elif return_image_latents or (latents is None and not is_strength_max):
#         image = image.to(device=device, dtype=dtype)
#         image_latents = _encode_vae_image(image=image, generator=generator)
#         image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

#     if latents is None and add_noise:
#         noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
#         # if strength is 1. then initialise the latents to noise, else initial to image + noise
#         latents = noise if is_strength_max else model.scheduler.add_noise(image_latents, noise, timestep)
#         # if pure noise then scale the initial latents by the  Scheduler's init sigma
#         latents = latents * model.scheduler.init_noise_sigma if is_strength_max else latents
#     elif add_noise:
#         noise = latents.to(device)
#         latents = noise * model.scheduler.init_noise_sigma
#     else:
#         noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
#         latents = image_latents.to(device)

#     outputs = (latents,)

#     if return_noise:
#         outputs += (noise,)

#     if return_image_latents:
#         outputs += (image_latents,)

#     return outputs



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
        if args.tracker_project_name:
            tracker_config = dict(vars(args))
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb": {"entity": args.tracker_entity}})


    test_dataset = UserDataDataset(
        dataroot_path=args.dataroot,
        phase="test",
        order="paired",
        size=(args.height, args.width)
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    model = ModelContainer(args,accelerator)
    
    unet_temp = UNet2DConditionModel_tryon.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    
    log_validation(unet_temp, model, args, accelerator, torch.float16, "old_model", test_dataloader)

    del unet_temp
    gc.collect()
    torch.cuda.empty_cache()
    
    unet = UNet2DConditionModel_tryon.from_pretrained(
        args.pretrained_nonfreeze_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant,
        encoder_hid_dim_type="ip_image_proj",
        encoder_hid_dim=1280,
        low_cpu_mem_usage=False
    )
    
    load_model_with_zeroed_mismatched_keys(unet, osp.join(args.pretrained_nonfreeze_model_name_or_path,"unet", "diffusion_pytorch_model.safetensors"))
    def replace_first_conv_layer(unet_model, new_in_channels):
        # Access the first convolutional layer
        # This example assumes the first conv layer is directly an attribute of the model
        # Adjust the attribute access based on your model's structure
        original_first_conv = unet_model.conv_in
        
        if(original_first_conv == new_in_channels):
            return
        
        # Create a new Conv2d layer with the desired number of input channels
        # and the same parameters as the original layer
        new_first_conv = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            padding=1,
        )
        
        # Zero-initialize the weights of the new convolutional layer
        new_first_conv.weight.data.zero_()

        # Copy the bias from the original convolutional layer to the new layer
        new_first_conv.bias.data = original_first_conv.bias.data.clone()
        
        new_first_conv.weight.data[:, :original_first_conv.in_channels] = original_first_conv.weight.data
        
        # Replace the original first conv layer with the new one
        return new_first_conv

    new_in_channel = 13
    unet.conv_in = replace_first_conv_layer(unet, new_in_channel)  #replace the conv in layer from 4 to 8 to make sd15 match with new input dims
    unet.config['in_channels'] = new_in_channel
    unet.config.in_channels = new_in_channel
    
    ################################DEBUG############################################
    # unet_com = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    # )
    # print("Tryon",unet)
    # print("Compare", unet_com)
    ################################DEBUG############################################
    

    # Freeze vae and text encoders.
    model.vae.requires_grad_(False)
    model.text_encoder_one.requires_grad_(False)
    model.text_encoder_two.requires_grad_(False)
    model.image_encoder.requires_grad_(False)
    model.ref_unet.requires_grad_(False)
    # Set unet as trainable.
    unet.train()

    # os._exit(os.EX_OK)
    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    model.vae.to(accelerator.device, dtype=torch.float32)
    model.text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    model.text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    model.ref_unet.to(accelerator.device, dtype=weight_dtype)
    model.image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    # if args.enable_npu_flash_attention:
    #     if is_torch_npu_available():
    #         logger.info("npu flash attention enabled.")
    #         unet.enable_npu_flash_attention()
    #     else:
    #         raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")
    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warning(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # os._exit(os.EX_OK)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataroot is None:
        assert "Please provide correct data root"
    # train_dataset = CPDataset(args.dataroot, args.resolution, mode="train", data_list=args.train_data_list)
    validation_dataset = UserDataDataset(
        dataroot_path=args.dataroot,
        phase="test",
        order="paired",
        size=(args.height, args.width)
    )
    
    train_dataset = UserDataDataset(
        dataroot_path=args.dataroot,
        phase="train",
        order="paired",
        size=(args.height, args.width)
    )
    
    # os._exit(os.EX_OK)
    
    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [model.text_encoder_one, model.text_encoder_two]
    tokenizers = [model.tokenizer_one, model.tokenizer_two]
    
    # del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
    # del text_encoders, tokenizers, vae
    gc.collect()
    torch.cuda.empty_cache()

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # os._exit(os.EX_OK)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, model.ref_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, model.ref_unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # put untrain modules into freeze

    # developing log
    # print(f"the in channel number: {unet.config.in_channels}")
    log_validation(unet, model, args, accelerator, weight_dtype, "pre_train", validation_dataloader)
    
    # os._exit(os.EX_OK)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # generator = torch.Generator(accelerator.device).manual_seed(args.seed) if args.seed is not None else None


    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                check_sample_elements(batch)
                img_emb_list = []
                for i in range(batch['cloth'].shape[0]):
                    img_emb_list.append(batch['cloth'][i])
                
                prompt = batch["caption"]
                
                target_size = batch['image'].shape[2:]
                batch['pose_img'] = F.interpolate(batch['pose_img'], size=target_size, mode='bilinear', align_corners=False)
                batch['cloth_pure'] = F.interpolate(batch['cloth_pure'], size=target_size, mode='bilinear', align_corners=False)

                pose_img = batch["pose_img"]
                cloth = batch["cloth_pure"]

                num_prompts = batch['cloth'].shape[0]                                        

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                
                image_embeds = torch.cat(img_emb_list,dim=0)

                output_hidden_state = not isinstance(unet.encoder_hid_proj, ImageProjection)
                image_embeds, _ = encode_image(
                    model, image_embeds, accelerator.device, 1, output_hidden_state
                )
                # print(f"image_embeds type {image_embeds.dtype}")
                # TODO: This is the temp solution of dtype assigment
                tmp_dtype = next(unet.parameters()).dtype
                image_embeds = unet.encoder_hid_proj(image_embeds.to(tmp_dtype))
                # print(f"image_embeds shape: {image_embeds.shape}")
                # os._exit(os.EX_OK)
                            
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                ) = encode_prompt(
                    prompt,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                )
                
                # print(f"prompt_embeds shape: {prompt_embeds.shape}")
                # os._exit(os.EX_OK)
                
                prompt_cloth = batch["caption_cloth"]
                if not isinstance(prompt_cloth, List):
                    prompt_cloth = [prompt_cloth] * num_prompts
                
                (
                    prompt_embeds_c,
                    _,
                ) = encode_prompt(
                    prompt_cloth,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                )
                # is_0_1 = is_in_range(batch["image"], 0 , 1)
                # print(f"is_0_1: {is_0_1}")
                init_image = (batch["image"] + 1.0) / 2.0
                init_image = model.image_processor.preprocess(
                    init_image, height=args.height, width=args.width, crops_coords=None, resize_mode="default"
                )
                mask = model.mask_processor.preprocess(
                    batch["inpaint_mask"], height=args.height, width=args.width, crops_coords=None, resize_mode="default"
                )
                if init_image.shape[1] == 4:
                    # if images are in latent space, we can't mask it
                    masked_image = None
                else:
                    masked_image = init_image * (mask < 0.5)
                
                model_input = compute_vae_encodings(init_image, model.vae)
                # print(f"model_input: {model_input.shape}")
                mask = torch.nn.functional.interpolate(
                    mask, size=(args.height // model.vae_scale_factor, args.width // model.vae_scale_factor)
                )
                mask = mask.to(device=accelerator.device, dtype=weight_dtype)
                # print(f"mask: {mask.shape}")
                masked_image_latents = compute_vae_encodings(masked_image, model.vae)
                # print(f"masked_image_latents: {masked_image_latents.shape}")
                
                pose_img_latents = compute_vae_encodings(pose_img, model.vae)
                # print(f"pose_img_latents: {pose_img_latents.shape}")
                
                cloth_latents = compute_vae_encodings(cloth, model.vae)
                # print(f"cloth_latents: {cloth_latents.shape}")
                # os._exit(os.EX_OK)
                reconstruct_model_input = model.reconstruct_vae_img(model_input)
                reconstruct_pose_image = model.reconstruct_vae_img(masked_image_latents)
                reconstruct_cloth_image = model.reconstruct_vae_img(cloth_latents)
                # print(f"reconstruct_model_input, {reconstruct_model_input}")
                # print(f"reconstruct_pose_image, {reconstruct_pose_image}")
                # print(f"reconstruct_cloth_image, {reconstruct_cloth_image}")

                # formatted_images = []
                # formatted_images.append(wandb.Image(reconstruct_model_input[0], caption="reconstruct_model_input"))
                # formatted_images.append(wandb.Image(reconstruct_pose_image[0], caption="reconstruct_pose_image"))
                # formatted_images.append(wandb.Image(reconstruct_cloth_image[0], caption="reconstruct_cloth_image"))
                # accelerator.log({"trainning internal image input": formatted_images})
                
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, model.noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # print(f"timesteps of training: {timesteps}")
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # Sample noise that we'll add to the latents

                noisy_model_input = model.noise_scheduler.add_noise(model_input, noise, timesteps)
                # print(f"noisy_model_input shape: {noisy_model_input.shape}")
                # os._exit(os.EX_OK)
                
                latent_model_input = torch.cat([noisy_model_input, mask, masked_image_latents,pose_img_latents], dim=1)
                # print(f"latent_model_input shape: {latent_model_input.shape}")
                
                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.width)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids
                
                original_size = (args.height, args.width)
                crops_coords_top_left = (0, 0)
                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip([original_size] * args.train_batch_size, [crops_coords_top_left] * args.train_batch_size)]
                )
                
                add_text_embeds = pooled_prompt_embeds
                # print(f"add_text_embeds shape: {add_text_embeds.shape}")
                # print(f"add_time_ids shape: {add_time_ids.shape}")
                
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                added_cond_kwargs.update({"image_embeds": image_embeds})
                
                # Predict the noise residual
                down,reference_features = model.ref_unet(
                    cloth_latents,
                    timesteps,
                    prompt_embeds_c,
                    return_dict=False
                )
                # print(f"reference_features: {reference_features}")
                # os._exit(os.EX_OK)
                
                reference_features = list(reference_features)
                
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    garment_features=reference_features,
                )[0]
                
                # # latents = model.noise_scheduler.step(model_pred, timesteps, latent_model_input, return_dict=False)[0]
                # reconstruct_latent_image = model.reconstruct_vae_img(model_pred)
                # # print(f"reconstruct_latent_image: {reconstruct_latent_image}")
                # formatted_images = []
                # formatted_images.append(wandb.Image(reconstruct_latent_image[0], caption="reconstruct_latent_image"))
                # accelerator.log({"trainning internal image output": formatted_images})
                
                # os._exit(os.EX_OK)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    model.noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if model.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif model.noise_scheduler.config.prediction_type == "v_prediction":
                    target = model.noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif model.noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {model.noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(model.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if model.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif model.noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    if global_step % args.validation_steps == 0:
                        log_validation(unet, model, args, accelerator, weight_dtype, "during_train", validation_dataloader)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unwrap_model(unet)
    #     if args.use_ema:
    #         ema_unet.copy_to(unet.parameters())

    #     # Serialize pipeline.
    #     vae = AutoencoderKL.from_pretrained(
    #         vae_path,
    #         subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #     )
    #     pipeline = StableDiffusionXLPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         unet=unet,
    #         vae=vae,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #     )
    #     if args.prediction_type is not None:
    #         scheduler_args = {"prediction_type": args.prediction_type}
    #         pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    #     pipeline.save_pretrained(args.output_dir)

    #     # run inference
    #     images = []
    #     if args.validation_prompt and args.num_validation_images > 0:
    #         pipeline = pipeline.to(accelerator.device)
    #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    #         with autocast_ctx:
    #             images = [
    #                 pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
    #                 for _ in range(args.num_validation_images)
    #             ]

    #         for tracker in accelerator.trackers:
    #             if tracker.name == "tensorboard":
    #                 np_images = np.stack([np.asarray(img) for img in images])
    #                 tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #             if tracker.name == "wandb":
    #                 tracker.log(
    #                     {
    #                         "test": [
    #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                             for i, image in enumerate(images)
    #                         ]
    #                     }
    #                 )

    #     if args.push_to_hub:
    #         save_model_card(
    #             repo_id=repo_id,
    #             images=images,
    #             validation_prompt=args.validation_prompt,
    #             base_model=args.pretrained_model_name_or_path,
    #             dataset_name=args.dataset_name,
    #             repo_folder=args.output_dir,
    #             vae_path=args.pretrained_vae_model_name_or_path,
    #         )
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

    accelerator.end_training()


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = parse_args()
    main(args)
    end_time = time.perf_counter()
    print(f"执行时间：{end_time - start_time}秒")
