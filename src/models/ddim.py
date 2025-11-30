import copy
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule, build_norm_layer, build_upsample_layer
from typing import Union, Dict, Tuple, Optional

from src.models.scheduling_ddim import DDIMScheduler
from src.models.hahi import HAHIHeteroNeck

class DDIMDepthHead(BaseModule):
    """DDIM Depth Refinement Head. This head refines an initial depth estimate using a Denoising Diffusion Implicit Model (DDIM) conditioned on image features."""
    def __init__(self, inference_steps=20, num_train_timesteps=1000):
        super().__init__()
        self.depth_transform = CNNDepthTransform(hidden=16, eps=1e-6)
        self.model = DenoisingDiffusionModel(channels_in=256, channels_noise=16)
        self.diffusion_inference_steps = inference_steps 
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.denoising_pipeline = DDIMDepthPipeline(self.model, self.scheduler)
        self.hahineck = HAHIHeteroNeck(in_channels=[192, 384, 768, 1536], out_channels=[192, 384, 768, 1536], embedding_dim=512, positional_encoding=dict(type='SinePositionalEncoding', num_feats=256), scales=[1, 1, 1, 1], cross_att=False, self_att=False, num_points=8)
        
        self.conv_lateral = ModuleList()
        self.conv_up = ModuleList()
        in_channels = [192, 384, 768, 1536]
        for i in range(len(in_channels)):
            self.conv_lateral.append(nn.Sequential(nn.Conv2d(in_channels[i], 256, 3, 1, 1, bias=False), build_norm_layer(dict(type='BN'), 256)[1], nn.ReLU(True)))
            if i != 0:
                self.conv_up.append(nn.Sequential(build_upsample_layer(dict(type='deconv', bias=False), in_channels=256, out_channels=256, kernel_size=2, stride=2), build_norm_layer(dict(type='BN'), 256)[1], nn.ReLU(True)))
        

    def forward(self, features, gt_depth_map):
        """Forward function for DDIM depth refinement head."""
        # Transform gt depth map from metric space (meters) to normalized latent space
        gt_depth_latent = self.depth_transform.t(gt_depth_map)
       
        # Enhance Features with HAHI Attention
        enhanced_features = self.hahineck(features)
        
        # Build Feature Pyramid Network (FPN)        
        fpn_features = None  # Will store fused informations from all 4 Swin stages
        for scale_idx in range(len(enhanced_features)):
            current_scale_features = enhanced_features[len(enhanced_features) - scale_idx - 1] # Process scales in reverse order from deepest (H/32) to shallowest (H/4)
            # Each scale has different channel dimensions (192/384/768/1536), so we project all to 256 channels in order to create a common feature space for fusion
            fpn_features = self.conv_lateral[len(enhanced_features) - scale_idx - 1](current_scale_features)
            if scale_idx > 0:                
                upsampled_prev_features = self.conv_up[len(enhanced_features) - scale_idx - 1](previous_fpn_features) # Upsample previous FPN features (e.g., from H/8 to H/4)
                upsampled_prev_features = nn.functional.adaptive_avg_pool2d(upsampled_prev_features, output_size=fpn_features.shape[-2:]) # Adaptive pool to ensure exact spatial match (handles any size mismatch)
                fpn_features = fpn_features + upsampled_prev_features # Add upsampled features to current scale (skip connection)
            previous_fpn_features = fpn_features # Store current FPN features for next iteration's upsampling
        
        # Run Denoising Diffusion Implicit Model (DDIM) to predict depth conditioned on FPN features
        refined_depth_latent = self.denoising_pipeline(gt_depth_latent=gt_depth_latent, conditioning_features=fpn_features, num_inference_steps=self.diffusion_inference_steps)
        
        # Convert from normalized latent space back to metric depth (meters)
        refined_depth = self.depth_transform.inv_t(refined_depth_latent)
        
        # Calculate denoising diffusion loss to train the model
        ddim_loss = self.ddim_loss(conditioning_features=fpn_features, depth_latent_clean=refined_depth_latent)

        output = {'pred': refined_depth, 'ddim_loss': ddim_loss}

        return output

    def ddim_loss(self, conditioning_features, depth_latent_clean):
        """
        This implements the DDIM training objective: teach the model to predict noise that was added to clean depth latents. 
        The model learns to denoise by minimizing the MSE between predicted and actual noise.
        """
        # Create noise to corrupt the clean depth latent. This noise will be what the model learns to predict
        random_noise = torch.randn(depth_latent_clean.shape).to(depth_latent_clean.device)

        # Sample a random timestep for each sample in the batch (timestep controls noise level)
        random_timesteps = torch.randint(low=0, high=self.scheduler.num_train_timesteps, size=(depth_latent_clean.shape[0],), device=depth_latent_clean.device).long()

        # Corrupt the clean depth by adding noise according to the diffusion schedule (this is the forward diffusion process)
        noisy_depth_latent = self.scheduler.add_noise(depth_latent_clean, random_noise, random_timesteps)

        # The model predicts the noise present in the noisy depth latent conditioned on image features
        noise_pred = self.model(noisy_depth_latent, random_timesteps, conditioning_features)

        # Compute MSE loss between predicted noise and actual noise
        loss = F.mse_loss(noise_pred, random_noise)

        return loss


class DDIMDepthPipeline:
    """
    Denoising Diffusion Implicit Model (DDIM) pipeline for depth estimation.
    This pipeline implements the DDIM sampling algorithm which iteratively denoises a random noise tensor to produce a clean depth map (in latent space).
    """
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(self, gt_depth_latent, conditioning_features, eta: float = 0.0, num_inference_steps: int = 50):
        # Create random Gaussian noise in latent space as the starting point. This noise will be progressively denoised into a clean depth map
        depth_latent = torch.randn((gt_depth_latent.shape), generator=None, device=conditioning_features.device, dtype=conditioning_features.dtype)
        
        # Configure the timesteps for the denoising process
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Progressively denoise the image over multiple timesteps. Each iteration removes a portion of the noise
        for timestep in self.scheduler.timesteps:
            # The model predicts what noise is present in the current noisy image conditioned on the features from the RGB image
            predicted_noise = self.model(depth_latent, timestep.to(conditioning_features.device), conditioning_features)
            
            # Use DDIM scheduler to compute the previous (less noisy) latent
            depth_latent = self.scheduler.step(predicted_noise, timestep, depth_latent, eta=eta, use_clipped_model_output=True, generator=None)['prev_sample']

        return depth_latent



class UpSample_add(nn.Sequential):
    '''Fusion module from Adabins'''
    def __init__(self, skip_input, output_features):
        super(UpSample_add, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, act_cfg=None)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, act_cfg=None)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x + concat_with))


class DenoisingDiffusionModel(BaseModule):
    """
    CNN for predicting noise in DDIM diffusion.
    This model takes noisy depth latents and predicts the noise that was added, conditioned on:
        1. Timestep t (how much noise is present)
        2. FPN features (scene context from RGB image)
    
    Args:
        channels_in (int): Number of input channels from FPN features (256)
        channels_noise (int): Number of channels in noisy depth latent (16)
    """
    def __init__(self, channels_in, channels_noise):
        super().__init__()

        # ========== Noise Embedding ==========
        # Process noisy depth latent to match FPN feature dimensions
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_in, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_in),
            nn.ReLU(True),
        )

        # ========== Feature Fusion Module ==========
        # Fuses FPN features (with timestep) and noisy depth embedding. This is the core computation that combines context and noise info
        self.upsample_fuse = UpSample_add(channels_in, channels_in)

        # ========== Timestep Embedding ==========
        # Learned embedding that encodes diffusion timestep information
        self.time_embedding = nn.Embedding(1280, channels_in)

        # ========== Noise Prediction Head ==========
        # Predict the noise that was added to create noisy_depth
        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_noise, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(True),
        )

    def forward(self, noisy_depth, timestep, conditioning_features):
        conditioning_features = conditioning_features + self.time_embedding(timestep)[..., None, None]

        fused_features = self.upsample_fuse(conditioning_features, self.noise_embedding(noisy_depth))

        return self.pred(fused_features)

class CNNDepthTransform(BaseModule):
    """
    Learned CNN-based depth transform with encoder-decoder architecture.
    Transforms depth from metric space (meters) to normalized latent space using a small convolutional network. 
    The inverse transform converts back to metric depth space.
    """
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        # ========== Encoder: Depth -> Latent Space ==========
        self.conv_transform = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, hidden, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.LeakyReLU(0.2, inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hidden)
            ),

            nn.Tanh()
        )
        # ========== Decoder: Latent Space -> Depth ==========
        self.conv_inv_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Sequential(nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1, bias=True)),

            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1