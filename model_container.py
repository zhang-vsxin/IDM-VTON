import torch

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoTokenizer, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPImageProcessor

from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

class ModelContainer:
    def __init__(self, args, accelerator, **kwargs):
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     subfolder="unet",
        #     torch_dtype=torch.float16,
        # )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.ref_unet = UNet2DConditionModel_ref.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.feature_extractor = CLIPImageProcessor()

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

    def reconstruct_vae_img(self, latent_img, output_type="pil"):
        """
        Reconstructs the original pose image from the latent representation and reverses the concatenation operation if needed.

        Args:
            latent_img (torch.Tensor): The latent representation of the pose image.
            vae (VAE): The VAE model used for encoding and decoding.
            device (torch.device): The device to perform computations on.
            dtype (torch.dtype): The data type of the original embeddings.
            scaling_factor (float): The scaling factor used during encoding.
            do_classifier_free_guidance (bool): Flag indicating whether classifier-free guidance was used.

        Returns:
            torch.Tensor: The reconstructed original pose image.
        """
        with torch.no_grad():
        # Reverse the scaling factor
            latent_img = latent_img / self.vae.config.scaling_factor
            
            # Decode the latent representation back to image space
            reconstructed_pose_img = self.vae.decode(latent_img, return_dict=False)[0]

            reconstructed_pose_img = self.image_processor.postprocess(reconstructed_pose_img, output_type=output_type)
            
            return reconstructed_pose_img

