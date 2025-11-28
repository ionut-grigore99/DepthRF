import torch
import numpy as np
import numba 
import collections
import cv2
import shutil
import os
from numba import prange
import copy
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmcv.utils import Registry
from torch import nn
import torch.nn.functional as F

DEPTH_TRANSFORM = Registry('depth_transforms', )

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*", "*weights*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    # Get the src directory (parent of utils.py)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy only src/ folder to backup location
    shutil.copytree(src_dir, backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)

'''
Propagates depth values along a single directional line (row or column) to fill missing depth pixels.
'''
@numba.njit() # decorator used to speed up numerical computations by compiling a function to machine code using Numba, a Just-In-Time (JIT) compiler for Python.
def simple_depth_completion_inner(canvas, distance_record, start, step):
    INF = 1e8
    rows, cols = canvas.shape 
    current_pos = start
    # prev_depth = -1.0 
    prev_depth = 0
    prev_distance = INF
    step_length = np.sqrt(np.sum(step ** 2))
    while (
        current_pos[0] >= 0 and 
        current_pos[0] < rows and 
        current_pos[1] >= 0 and 
        current_pos[1] < cols 
    ):
        index = (current_pos[0], current_pos[1])
        # if canvas[index] == -1: 
        if canvas[index] == 0: 
            canvas[index] = prev_depth 
            distance_record[index] = prev_distance 
        else:  # != -1
            distance1 = distance_record[index] 
            distance2 = prev_distance
            if distance1 > distance2: 
                distance_record[index] = distance2
                canvas[index] = prev_depth 
            prev_depth = canvas[index]
            prev_distance = distance_record[index]
        prev_distance += step_length
        current_pos += step


'''
Fills holes and missing regions in sparse depth maps using bidirectional propagation from all 4 directions:
 - vertical: top-to-bottom and bottom-to-top
 - horizontal: left-to-right and right-to-left
'''
@numba.njit(parallel=True) # telling Numba to automatically parallelize certain operations inside the function — especially loops — to run across multiple CPU cores.
def simple_depth_completion(sparse_depth):
    INF = 1e8
    rows, cols = sparse_depth.shape
    canvas = np.copy(sparse_depth)
    distance_record = np.zeros((rows, cols), dtype=np.float32)
    for c in prange(cols):
        simple_depth_completion_inner(canvas, distance_record, np.array([0, c]), np.array([1, 0]))
        simple_depth_completion_inner(canvas, distance_record, np.array([rows - 1, c]), np.array([-1, 0]))
    for r in prange(rows):
        simple_depth_completion_inner(canvas, distance_record, np.array([r, 0]), np.array([0, 1]))
        simple_depth_completion_inner(canvas, distance_record, np.array([r, cols - 1]), np.array([0, -1]))
    return canvas, distance_record


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


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5, extrapolate=False, blur_type='bilateral', blur_kernel_size=5):
    """Fast, in-place depth completion.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE
    Returns:
        depth_map: dense depth map
    """
    depth_map = depth_map.copy()

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, blur_kernel_size, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (blur_kernel_size, blur_kernel_size), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict
    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map).copy()

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out


def convert_depth_map_to_points(depth, input_size, downsample, rots, trans, intrins, post_rots, post_trans, decoration_img=None, return_batch_idx=True):
    bs, n_cam, n_depth, h, w = depth.shape
    frustum = _create_frustum(depth, input_size, downsample)
    geom = get_geometry(frustum, rots, trans, intrins, post_rots, post_trans)

    if decoration_img is not None: 
        geom = _decorate_points(geom, decoration_img)
    
    if return_batch_idx:
        geom = geom.view(-1, geom.shape[-1])

        batch_ix = torch.cat([torch.full([geom.shape[0] // bs, 1], ix, device=depth.device, dtype=torch.long) for ix in range(bs)])
        return geom, batch_ix
    else: 
        geom = geom.view(bs, -1, geom.shape[-1])
        return geom 


class BaseDepthRefine(BaseModule):
    def __init__(self, in_channels, detach_fp=False, blur_depth_head=True, depth_embed_dim=16, depth_feature_dim=16, loss_cfgs=[], depth_transform_cfg=dict(type='ReciprocalDepthTransform'), upsample_cfg=dict(type='deconv', bias=False), norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        depth_feature_dim=16
        self.init_cfg = init_cfg
        self.detach_fp = detach_fp
        self.depth_embed_dim = depth_embed_dim
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.loss_cfgs = loss_cfgs
        self.conv_lateral = ModuleList()
        self.conv_up = ModuleList()
        for i in range(len(in_channels)):
            self.conv_lateral.append(nn.Sequential(nn.Conv2d(depth_feature_dim + 1, depth_embed_dim, 3, 1, 1, bias=False), build_norm_layer(norm_cfg, depth_embed_dim)[1], nn.ReLU(True)))

            if i != 0:
                self.conv_up.append(nn.Sequential(build_upsample_layer(upsample_cfg, in_channels=in_channels[i] + depth_embed_dim, out_channels=in_channels[i - 1] + depth_embed_dim, kernel_size=2, stride=2), build_norm_layer(norm_cfg, in_channels[i - 1] + depth_embed_dim)[1], nn.ReLU(True)))

        if blur_depth_head:
            self.blur_depth_head = nn.Sequential(nn.Conv2d(in_channels[0] + depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False), build_norm_layer(norm_cfg, depth_embed_dim)[1], nn.ReLU(True), nn.Conv2d(depth_embed_dim, 1, 3, 1, 1, ), nn.Sigmoid())

        self.weight_head = nn.Sequential(nn.Conv2d(in_channels[0] + depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False), build_norm_layer(norm_cfg, depth_embed_dim)[1], nn.ReLU(True), nn.Conv2d(depth_embed_dim, 1, 3, 1, 1))

    def init_weights(self):
        super().init_weights()

    def forward(self, fp, depth_map, depth_mask):
        '''
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        depth_mask: Tensor with shape bs, 1, h, w
        '''

        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                fp = [it for it in fp]
                for i in self.detach_fp:
                    fp[i] = fp[i].detach()
            else:
                fp = [it.detach() for it in fp]

        depth_map_t = self.depth_transform.t(depth_map)
        depth = torch.cat((depth_map_t, depth_mask), dim=1)  # bs, 2, h, w
        for i in range(len(fp)):
            f = fp[len(fp) - i - 1]
            depth_down = nn.functional.adaptive_avg_pool2d(depth, output_size=f.shape[-2:])
            depth_embed = self.conv_lateral[len(fp) - i - 1](depth_down)
            x = torch.cat((f, depth_embed), axis=1)
            if i > 0:
                x = x + self.conv_up[len(fp) - i - 1](pre_x)
            pre_x = x

        if hasattr(self, 'blur_depth_head'):
            blur_depth = self.blur_depth_head(x)
            blur_depth = self.depth_transform.inv_t(blur_depth)
        else:
            blur_depth = depth_map

        depth_weight = self.weight_head(x).sigmoid().clamp(1e-3, 1 - 1e-3)
        return x, blur_depth, depth_weight

    def loss(self, pred_depth, gt_depth, pred_uncertainty=None, weight_map=None, instance_masks=None, image=None, **kwargs):
        loss_dict = {}
        for loss_cfg in self.loss_cfgs:
            loss_fnc_name = loss_cfg['loss_func']
            loss_key = loss_cfg['name']
            if loss_fnc_name not in depth_loss_dict:
                continue
            loss = depth_loss_dict[loss_fnc_name](pred_depth=pred_depth, pred_uncertainty=pred_uncertainty, gt_depth=gt_depth, weight_map=weight_map, instance_masks=instance_masks, image=image, **loss_cfg, **kwargs)
            loss_dict[loss_key] = loss
        return loss_dict


def l1_depth_loss(pred_depth, gt_depth, pred_indices=None, gt_indices=None, weight=1., weight_map=None, **kwargs):
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)
    gt_mask = gt_depth >= 0.0001
    loss = (pred_depth - gt_depth).abs() * gt_mask
    if weight_map is not None:
        loss *= weight_map
    loss = loss.sum() / gt_mask.sum().clamp(1.)
    return loss * weight


def depth_smooth_loss(pred_depth, gt_depth, image, instance_masks, pred_indices=None, gt_indices=None, weight=1., eps=1e-6, **kwargs):
    if pred_indices is not None:
        pred_depth = pred_depth[..., pred_indices, :, :]
    if gt_indices is not None:
        gt_depth = gt_depth[..., gt_indices, :, :]
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)

    def try_resize(input_img, shape):
        if input_img.shape[-2:] != shape:
            old_shape = input_img.shape
            image_reshape = input_img.view(-1, 1, *old_shape[-2:])
            image_reshape = F.interpolate(image_reshape, shape)
            image_reshape = image_reshape.view(*old_shape[:-2], *shape)
            return image_reshape
        return input_img

    image = try_resize(image, pred_depth.shape[-2:])
    instance_masks = instance_masks.float()
    max_id = F.max_pool2d(instance_masks, 3, stride=1, padding=1)
    min_id = -F.max_pool2d(-instance_masks, 3, stride=1, padding=1)
    edge_masks = (max_id != min_id).float()
    edge_masks = F.adaptive_max_pool2d(edge_masks, output_size=pred_depth.shape[-2:])

    pred_depth = pred_depth * (1 - edge_masks) + pred_depth.detach() * edge_masks

    grad_depth_x = torch.abs(pred_depth[..., :-1] - pred_depth[..., 1:])
    grad_depth_y = torch.abs(pred_depth[..., :-1, :] - pred_depth[..., 1:, :])

    grad_img_x = torch.mean(torch.abs(image[..., :-1] - image[..., 1:]), -3, keepdim=False)
    grad_img_y = torch.mean(torch.abs(image[..., :-1, :] - image[..., 1:, :]), -3, keepdim=False)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    return (grad_depth_x.mean() + grad_depth_y.mean()) * weight


def shape_reg_loss(pred_depth, gt_depth, foreground_masks, gt_bboxes_3d, rots, trans, intrins, post_rots, post_trans, input_size, downsample, pred_indices=None, gt_indices=None, weight=1., eps=1e-6, max_distance=1, focus=False, **kwargs):
    if pred_indices is not None:
        pred_depth = pred_depth[..., pred_indices, :, :]
    if gt_indices is not None:
        gt_depth = gt_depth[..., gt_indices, :, :]
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)

    # 1. convert depth_map to xyz_map
    xyz, _ = convert_depth_map_to_points(pred_depth.unsqueeze(2), input_size, downsample, rots, trans, intrins, post_rots, post_trans)
    xyz = xyz.view(*pred_depth.shape, 3)

    batch_size = len(gt_bboxes_3d)
    losses = [_shape_reg_loss_per_batch(xyz[b], gt_depth[b], foreground_masks[b], gt_bboxes_3d[b], max_distance=max_distance) for b in range(batch_size)]

    if focus:
        losses = [it[it > 0] for it in losses]
    return torch.cat(losses).mean() * weight


def _shape_reg_loss_per_batch(xyz, gt_depth, foreground_masks, gt_bboxes_3d, max_distance):
    gt_boxes = gt_bboxes_3d.tensor.to(xyz.device)
    cos_theta = torch.cos(gt_boxes[:, 6])
    sin_theta = torch.sin(gt_boxes[:, 6])
    zeros = sin_theta * 0.
    ones = zeros + 1.
    rot_mat = torch.stack([cos_theta, -sin_theta, zeros, sin_theta, cos_theta, zeros, zeros, zeros, ones], dim=-1).view(-1, 3, 3).to(xyz.device)
    box_centers = gt_boxes[:, :3]  # n_box, 3
    box_centers[:, 2] += gt_boxes[:, 5] / 2
    box_sizes = gt_boxes[:, 3:6].to(xyz.device)  # n_box, 3

    foreground_masks = _try_resize(foreground_masks.float(), gt_depth.shape[-2:], mode='nearest')
    xyz = xyz[foreground_masks > 0.5]  # n_pts, 3
    xyz_per_box = xyz.view(-1, 1, 3) - box_centers.unsqueeze(0)  # n_pts, n, 3
    xyz_per_box = xyz_per_box.unsqueeze(-2) @ rot_mat.permute(0, 2, 1)
    xyz_per_box = xyz_per_box.squeeze(-2)  # n_pts, n_box, 3

    loss = torch.min(torch.mean(torch.relu(xyz_per_box.abs() - box_sizes), dim=-1), dim=1)[0]  # n_pts

    return loss


depth_loss_dict = {'l1_depth_loss': l1_depth_loss, 'depth_smooth_loss': depth_smooth_loss, 'shape_reg_loss': shape_reg_loss}


def _try_resize(input_img, shape, mode='bilinear'):
    if input_img.shape[-2:] != shape:
        old_shape = input_img.shape
        image_reshape = input_img.view(-1, 1, *old_shape[-2:])
        image_reshape = F.interpolate(image_reshape, shape, mode=mode)
        image_reshape = image_reshape.view(*old_shape[:-2], *shape)
        return image_reshape
    return input_img

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, 'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers
    
@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsampling(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        self.conv_transform = nn.Sequential(conv_bn_relu(1, hidden, 3, 2, 1), conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False), nn.Tanh())
        self.conv_inv_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False),
            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsampling1x1(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        self.conv_transform = nn.Sequential(nn.Conv2d(1, hidden, 1, 1, 0, bias=False), nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv_inv_transform = nn.Sequential(nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False), nn.Sigmoid())
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsamplingX4(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        self.conv_transform = nn.Sequential(conv_bn_relu(1, hidden, 3, 2, 1), conv_bn_relu(hidden, hidden, 3, 2, 1), conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False), nn.Tanh())
        self.conv_inv_transform = nn.Sequential(nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1), nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False), nn.Sigmoid())
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransform(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        self.conv_transform = nn.Sequential(conv_bn_relu(1, hidden, 3, 1, 1), conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False), nn.Tanh())
        self.conv_inv_transform = nn.Sequential(conv_bn_relu(hidden, hidden, 3, 1, 1), conv_bn_relu(hidden, 1, 3, 1, 1, relu=False), nn.Sigmoid())
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class ReciprocalDepthTransform(object):
    def __init__(self, linear=(1, 0), eps=1e-6):
        self.linear = linear
        self.eps = eps

    def t(self, depth):
        '''
        return: transformed depth value in range (0, 1],
        '''
        return self.linear[0] / (1 + depth.clamp(0.)).clamp(self.eps) + self.linear[1]

    def inv_t(self, value):
        return (self.linear[0] / (value - self.linear[1]).clamp(self.eps) - 1)


@DEPTH_TRANSFORM.register_module()
class ReciprocalDepthTransformII(object):
    def __init__(self, min_depth=0.5):
        self.min_depth = min_depth

    def t(self, depth):
        return self.min_depth / depth.clamp(self.min_depth)

    def inv_t(self, value):
        return self.min_depth / value

