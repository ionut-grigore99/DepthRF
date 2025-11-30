import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np

from src.models.swin import swin_large_naive_l4w722422k
from src.models.ddim import DDIMDepthHead


class DiffusionDepth(nn.Module):
    def __init__(self, conf):
        super(DiffusionDepth, self).__init__()
        self.conf = conf
        self.depth_backbone = swin_large_naive_l4w722422k(pretrained=conf['backbone_weights']) 
        self.depth_head = DDIMDepthHead(inference_steps=conf['inference_steps'], num_train_timesteps=conf['num_train_timesteps'])
      
    def forward(self, sample):
        """Forward training function. Returns a dict with depth prediction and DDIM loss.
        """
        rgb = sample['rgb']
        depth_gt = sample['depth_gt']
        
        features = self.depth_backbone(rgb)
        output_dict = self.depth_head(features, depth_gt)
        
        return output_dict

    def from_pretrained(self, weights_path: str):
        """Load model weights from a pre-trained file.
        Args:
            weights_path (str): Path to the pre-trained weights file.
        """
        assert os.path.exists(weights_path), "file not found: {}".format(weights_path)
        print(f"loading checkpoint from local path: {weights_path}")

        checkpoint = torch.load(weights_path)
        key_m, key_u = self.load_state_dict(checkpoint['net'], strict=True)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

        print(f"complete loading with parameters pretrain from {weights_path}")

if __name__ == '__main__':
    from src.conf import Config
    conf = Config().conf

    model = DiffusionDepth(conf)
    model = model.to(conf['device'])
    model.from_pretrained(weights_path=conf['model_weights'])
    model.eval()