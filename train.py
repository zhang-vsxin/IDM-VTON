import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer

from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from lpips import LPIPS
from pytorch_msssim import ssim
from transformers import CLIPProcessor, CLIPModel





weight_dtype = torch.float16

logger = get_logger(__name__, log_level="INFO")

#wandb.init(project="Ayna-Finetune")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for IDM-VTON.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="../models", required=False)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()
    #wandb.config.update(args)
    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images

class DeepFashionTrainDataset(data.Dataset):
    def __init__(self, dataroot_path: str, phase: Literal["train", "test"], order: Literal["paired", "unpaired"] = "paired", size: Tuple[int, int] = (512, 384)):
        super(DeepFashionTrainDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.toTensor = transforms.ToTensor()

        with open(os.path.join(dataroot_path,f"garment_desc_{phase}.txt"), "r") as file1:
            self.data = json.load(file1)

        self.order = order
        self.toTensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()

    def __getitem__(self, index):
        item=self.data[index]
        cloth = Image.open(os.path.join(self.dataroot,"garments", item["type"],item["img_fn"]))

        im_pil_big = Image.open(os.path.join(self.dataroot, "models", item["img_fn"])).resize((self.width, self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, "garments", item["type"],"mask_"+item["img_fn"])).resize((self.width, self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        mask = 1-mask
        im_mask = image * mask

        pose_img = Image.open(os.path.join(self.dataroot, "models", "pose_"+item["img_fn"]))
        pose_img = self.transform(pose_img)  # [-1,1]

        result = {}
        result["c_name"] = item["img_fn"]
        result["im_name"] = item["img_fn"]
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        result["inpaint_mask"] = 1-mask
        result["im_mask"] = im_mask
        result["caption_cloth"] = "a photo of " + item["prompt"]
        result["caption"] = "model is wearing a " + item["prompt"]
        
        resize_transform = transforms.Resize((1024, 768))
        pose_img_resized = resize_transform(pose_img.unsqueeze(0)).squeeze(0)
        result["pose_img"] = pose_img

        return result

    def __len__(self):
        return len(self.data)
    

def compute_metrics(real_images, generated_images, lpips_model, clip_model, clip_processor, device):
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)
    
    real_images_np = real_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    generated_images_np = generated_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Compute LPIPS score
    lpips_score = lpips_model(real_images, generated_images).mean().item()
    
    # Compute SSIM score
    ssim_score = ssim(real_images, generated_images, data_range=1.0).item()
    
    # Compute CLIP image similarity
    real_images_pil = [Image.fromarray((img * 255).astype('uint8')) for img in real_images_np]
    generated_images_pil = [Image.fromarray((img * 255).astype('uint8')) for img in generated_images_np]
    
    real_images_clip = clip_processor(images=real_images_pil, return_tensors="pt")['pixel_values'].to(device)
    generated_images_clip = clip_processor(images=generated_images_pil, return_tensors="pt")['pixel_values'].to(device)
    
    real_features = clip_model.get_image_features(real_images_clip).detach().cpu().numpy()
    generated_features = clip_model.get_image_features(generated_images_clip).detach().cpu().numpy()
    
    clip_similarity = np.mean(np.dot(real_features, generated_features.T).diagonal())
    
    return lpips_score, ssim_score, clip_similarity





def train(args, train_dataloader, model, optimizer, accelerator):
    global_step = 0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                img_emb_list = [batch['cloth'][i] for i in range(batch['cloth'].shape[0])]
                prompt = batch["caption"]

                num_prompts = batch['cloth'].shape[0]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, list):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, list):
                    negative_prompt = [negative_prompt] * num_prompts

                image_embeds = torch.cat(img_emb_list, dim=0)

                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = model.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)

                prompt = batch["caption_cloth"]
                if not isinstance(prompt, list):
                    prompt = [prompt] * num_prompts

                prompt_embeds_c = model.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

                generator = torch.Generator(model.device).manual_seed(args.seed) if args.seed is not None else None
                
                target_size = batch['image'].shape[2:]
                batch['pose_img_resized'] = F.interpolate(batch['pose_img'], size=target_size, mode='bilinear', align_corners=False)
                batch['cloth_pure_resized'] = F.interpolate(batch['cloth_pure'], size=target_size, mode='bilinear', align_corners=False)

                images = model(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=batch['pose_img_resized'].to(accelerator.device, dtype=weight_dtype),
                    text_embeds_cloth=prompt_embeds_c,
                    cloth=batch["cloth_pure_resized"].to(accelerator.device, dtype=weight_dtype),
                    mask_image=batch['inpaint_mask'].to(accelerator.device, dtype=weight_dtype),
                    image=(batch['image'].to(accelerator.device, dtype=weight_dtype) + 1.0) / 2.0,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    ip_adapter_image=image_embeds.to(accelerator.device, dtype=weight_dtype)
                )[0]

                images_tensor = torch.stack([transforms.ToTensor()(img).to(accelerator.device) for img in images]).requires_grad_()
                batch_image_tensor = batch['image'].to(accelerator.device)
                loss = F.mse_loss(images_tensor, batch_image_tensor)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # Compute and log metrics
                #lpips_score, ssim_score, clip_similarity = compute_metrics(batch_image_tensor, images_tensor, lpips_model, clip_model, clip_processor, device)
                #wandb.log({"Step": global_step, "Loss": loss.item(), "LPIPS": lpips_score, "SSIM": ssim_score, "CLIP Similarity": clip_similarity})
                print(f"loss:{loss.item()}")

                if global_step % args.save_interval == 0:
                    print("Global_step: ", global_step)
                    for i in range(len(images)):
                        x_sample = transforms.ToTensor()(images[i])
                        torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, f"step_{global_step}_{batch['im_name'][i]}"))
                        print("Images generated!")
                    # Log generated images to wandb
                    #wandb.log({f"Generated Images step {global_step}": [wandb.Image(img) for img in images]})

                global_step += 1


    accelerator.wait_for_everyone()
    accelerator.end_training()
    



def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=accelerator_project_config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype)
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet_encoder", torch_dtype=weight_dtype)
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=None, use_fast=False)

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)

    # Allow gradients for the models to be trained
    unet.requires_grad_(True)
    image_encoder.requires_grad_(False)
    
    # initialize_weights(unet)
    # initialize_weights(image_encoder)

    unet.to(accelerator.device, weight_dtype)
    # image_encoder.to(accelerator.device, weight_dtype)

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    train_dataset = DeepFashionTrainDataset(dataroot_path=args.data_dir, phase="train", order="paired", size=(args.height, args.width))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)

    model = TryonPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    model.unet_encoder = UNet_Encoder.to('cuda')
    
    # Collect parameters from unet and image_encoder
    params_to_optimize = unet.parameters()
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.learning_rate)
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    unet.enable_gradient_checkpointing()
    unet.train()
    train(args, train_dataloader, model, optimizer, accelerator)
    unet.save_pretrained(args.output_dir,  torch_dtype=weight_dtype)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"执行时间：{end_time - start_time}秒")
