import torch
import numpy as np
import numba 
import collections
import cv2
import shutil
import os
from numba import prange
import copy
from mmcv.runner import BaseModule
from torch import nn
import torch.nn.functional as F


'''
Function to backup source code during training
'''
def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(".git*", "*pycache*", "*weights*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Copy src/ folder to backup location
    shutil.copytree(src_dir, backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

'''
Function to convert input samples to NumPy arrays
'''
class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)

'''
The function creates a sparse depth map by randomly sampling a specified number of valid depth pixels from the dense ground truth depth map.
'''
def get_sparse_depth(depth, num_sparse_points):
    channel, height, width = depth.shape

    assert channel == 1

    idx_nonzero = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nonzero)
    idx_sample = torch.randperm(num_idx)[:num_sparse_points]

    idx_nonzero = idx_nonzero[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nonzero] = 1.0
    mask = mask.view((channel, height, width))

    depth_sparse = depth * mask.type_as(depth)

    return depth_sparse


'''
Function to resize tensors with optional warning for align_corners usage
'''
def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)




