import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.core import bbox3d2result
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.models import BACKBONES
from mmdet3d.models.builder import HEADS, build_loss
import mmcv
from collections import OrderedDict

from src.utils import fill_in_multiscale, fill_in_fast
from src.models.swin import swin_large_naive_l4w722422k
from src.models.ddim_depth_estimate_res_swin_addHAHI import DDIMDepthEstimate_Swin_ADDHAHI


@DETECTORS.register_module()
class DiffusionDepth(nn.Module):
    def __init__(self, conf, **kwargs):
        super(DiffusionDepth, self).__init__()
        self.conf = conf
        self.depth_backbone = swin_large_naive_l4w722422k(pretrained=conf['backbone_weights'])
        depth_head_cfg=dict(type='DDIMDepthEstimate_Swin_ADDHAHI',
                        in_channels=[64, 128, 256, 512],  # ResNet-18
                        inference_steps=conf['inference_steps'],
                        num_train_timesteps=conf['num_train_timesteps'],
                        depth_feature_dim=16, 
                        loss_cfgs=[dict(loss_func='l1_depth_loss', name='depth_loss', weight=0.2, pred_indices=0, gt_indices=0), dict(loss_func='l1_depth_loss', name='blur_depth_loss', weight=0.1, pred_indices=1, gt_indices=0)],
                        init_cfg=conf)
        self.depth_head = HEADS.build(depth_head_cfg)

    def extract_depth(self, img, depth_map, depth_mask, gt_depth_map, return_loss, img_metas, weight_map=None, instance_masks=None, **kwargs):
        B, C, imH, imW = img.shape
        img = img.view(B, C, imH, imW)
        depth_map = depth_map.view(B, 1, *depth_map.shape[-2:])
        gt_depth_map = gt_depth_map.view(B, 1, *depth_map.shape[-2:]) if gt_depth_map is not None else None
        weight_map = weight_map.view(B , *depth_map.shape[-2:]) if weight_map is not None else None
        instance_masks = instance_masks.view(B, 1, *instance_masks.shape[-2:]) if instance_masks is not None else None
        depth_mask = depth_mask.view(*depth_map.shape)
        fp = self.depth_backbone(img)
        ret = self.depth_head(fp, depth_map, depth_mask, gt_depth_map=gt_depth_map, return_loss=return_loss, weight_map=weight_map, instance_masks=instance_masks, image=img, **kwargs)

        return ret
      
    def forward(self, sample):
        """Forward training function.
        Args:
            sample containing 4 keys:
                rgb          - torch.Size([3, 3, W, H])
                depth_sparse - torch.Size([3, 1, W, H])
                depth_gt     - torch.Size([3, 1, W, H])
                K            - torch.Size([3, 4])

        Returns:
            output_dict: Losses of different branches.
        """
        img_inputs = sample['rgb']
        gt_depth_map = sample['depth_gt']
        sparse_depth = sample['depth_sparse']
        depth_mask = sample['valid_depth_mask'] 
        depth_map = sample['depth_filled'] 

        output_dict = self.extract_depth(img_inputs, depth_map, depth_mask, gt_depth_map, img_metas=None, return_loss=True, weight_map=None, instance_masks=None, sparse_depth=sparse_depth)
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