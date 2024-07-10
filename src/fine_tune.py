import argparse
import torch

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPImageProcessor
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

# 准备模型及各项参数
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "yisol/IDM-VTON",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="result",)
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--data_dir",type=str,default="/home/omnious/workspace/yisol/Dataset/zalando")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--test_batch_size", type=int, default=2,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()
    return args
# 从命令行获取参数
args = parse_args()
weight_dtype = torch.float16

# 实例化各个模型
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)
text_encoder_one = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
tokenizer_one = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

# 定义 accelerator 加速器
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    project_config=accelerator_project_config,
)
    
# 冻结不需要微调的层。要调的是unet模型（根据论文，它是tryon-net的decoder层）。
# 打开unet模型
unet.requires_grad_(True)
# 冻结其它层
vae.requires_grad_(False)
image_encoder.requires_grad_(False)
UNet_Encoder.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
UNet_Encoder.to(accelerator.device, weight_dtype)
unet.eval()
UNet_Encoder.eval()

pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    ).to(accelerator.device)
pipe.unet_encoder = UNet_Encoder


# 定义loss function（根据论文）

# 选择Adam优化器（根据论文）

# 引入训练数据