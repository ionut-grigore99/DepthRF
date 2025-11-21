"""
    KITTI Depth Prediction Dataset Loader.

    KITTI Depth Prediction json file has a following format:

{
    "train": [
        {
            "rgb": "train/2011_09_30_drive_0018_sync/image_03/data/0000002698.png",
            "depth": "train/2011_09_30_drive_0018_sync/proj_depth/velodyne_raw/image_03/0000002698.png",
            "gt": "train/2011_09_30_drive_0018_sync/proj_depth/groundtruth/image_03/0000002698.png",
            "K": "train/2011_09_30_drive_0018_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "val": [
        {
            "rgb": "val/2011_09_26_drive_0023_sync/image_03/data/0000000218.png",
            "depth": "val/2011_09_26_drive_0023_sync/proj_depth/velodyne_raw/image_03/0000000218.png",
            "gt": "val/2011_09_26_drive_0023_sync/proj_depth/groundtruth/image_03/0000000218.png",
            "K": "val/2011_09_26_drive_0023_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "test": [
        {
            "rgb": "depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000215_image_03.png",
            "depth": "depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0023_sync_velodyne_raw_0000000215_image_03.png",
            "gt": "depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0023_sync_groundtruth_depth_0000000215_image_03.png",
            "K": "depth_selection/val_selection_cropped/intrinsics/2011_09_26_drive_0023_sync_image_0000000215_image_03.txt"
        }, ...
    ]
}
"""


import os
import warnings
import numpy as np
import json
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import lovely_tensors as lt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.utils import get_sparse_depth, simple_depth_completion

warnings.filterwarnings("ignore", category=UserWarning)


def read_depth(file_name):
    # Loads depth map D from 16 bits png file as a numpy array, refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class KITTI(Dataset):
    def __init__(self, data_dir, split_json, use_aug, mode, num_sparse_points, test_crop, top_crop, patch_height, patch_width):
        super(KITTI, self).__init__()

        self.data_dir = data_dir # Root directory of KITTI dataset
        self.split_json = split_json # Path to the split JSON file
        self.use_aug = use_aug # Whether to use data augmentation
        self.mode = mode # 'train', 'val', or 'test'
        self.num_sparse_points = num_sparse_points # Number of sparse depth points to sample from the dense depth map
        self.test_crop = test_crop # Whether to apply cropping during testing
        self.top_crop = top_crop # Number of pixels to crop from the top of the image
        self.height = patch_height # Height of the cropped patch
        self.width = patch_width # Width of the cropped patch
        

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        with open(self.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)

        if self.use_aug and self.mode == 'train': # apply data augmentations only during training as flipping, rotation, color jittering and transformations such as resizing, cropping and normalization 
            # Top crop if needed
            if self.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth = TF.crop(depth, self.top_crop, 0, height - self.top_crop, width)
                gt = TF.crop(gt, self.top_crop, 0, height - self.top_crop, width)
                K[3] = K[3] - self.top_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                gt = TF.hflip(gt)
                K[2] = width - K[2]

            # Rotation
            rgb = TF.rotate(rgb, angle=degree, resample=Image.BICUBIC)
            depth = TF.rotate(depth, angle=degree, resample=Image.NEAREST)
            gt = TF.rotate(gt, angle=degree, resample=Image.NEAREST)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            # Resize
            rgb = TF.resize(rgb, scale, Image.BICUBIC)
            depth = TF.resize(depth, scale, Image.NEAREST)
            gt = TF.resize(gt, scale, Image.NEAREST)

            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

            # Crop
            width, height = rgb.size
          
            assert self.height <= height and self.width <= width, "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            depth = depth / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale
        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth = TF.crop(depth, self.top_crop, 0, height - self.top_crop, width)
                gt = TF.crop(gt, self.top_crop, 0, height - self.top_crop, width)
                K[3] = K[3] - self.top_crop

            # Crop
            width, height = rgb.size
            
            assert self.height <= height and self.width <= width, "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))
        else:
            if self.top_crop > 0 and self.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth = TF.crop(depth, self.top_crop, 0, height - self.top_crop, width)
                gt = TF.crop(gt, self.top_crop, 0, height - self.top_crop, width)
                K[3] = K[3] - self.top_crop

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))

        if self.num_sparse_points > 0:
            depth_sparse = self.get_sparse_depth(depth, self.num_sparse_points)
        else:
            depth_sparse = depth 
        
        # Create binary mask for valid depth pixels
        valid_depth_mask = (depth_sparse > 0)

        USE_DEPTH_COMPLETION = False  

        # Apply simple depth completion to fill gaps in sparse depth map
        if USE_DEPTH_COMPLETION:
            depth_filled = np.asarray(depth_sparse.squeeze(0), dtype=np.float32)  
            depth_filled, _ = simple_depth_completion(depth_filled)
            depth_filled = depth_filled[np.newaxis, ...]  
        else:
            depth_filled = depth_sparse  # Use sparse depth as-is

        output = {'rgb': rgb, 'depth_sparse': depth_sparse, 'depth_gt': gt, 'K': torch.Tensor(K), 'valid_depth_mask': valid_depth_mask, 'depth_filled': depth_filled}

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.data_dir, self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.data_dir, self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.data_dir, self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.data_dir, self.sample_list[idx]['K'])

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        if self.mode in ['train', 'val']:
            calib = read_calib_file(path_calib)
            if 'image_02' in path_rgb:
                K_cam = np.reshape(calib['P_rect_02'], (3, 4))
            elif 'image_03' in path_rgb:
                K_cam = np.reshape(calib['P_rect_03'], (3, 4))
            K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        else:
            f_calib = open(path_calib, 'r')
            K_cam = f_calib.readline().split(' ')
            f_calib.close()
            K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]), float(K_cam[5])]

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt, K

    '''
    The function creates a sparse depth map by randomly sampling a specified number of valid depth pixels from the dense ground truth depth map.
    '''
    def get_sparse_depth(self, depth, num_sparse_points):
        channel, height, width = depth.shape

        assert channel == 1

        idx_nnz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sparse_points]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        depth_sparse = depth * mask.type_as(depth)

        return depth_sparse

if __name__ == "__main__":
    lt.monkey_patch()  # Enable lovely_tensors for better debugging

    # Configurations
    data_dir = '/data/DepthRF/kitti/kitti_dp'
    split_json = '/home/ionut/DepthRF/src/datasets/kitti.json'
    use_aug = True
    mode = 'train'
    num_sparse_points = 0
    test_crop = False
    patch_height = 240
    patch_width = 1216
    top_crop = 100

    dataset = KITTI(data_dir, split_json, use_aug, mode, num_sparse_points, test_crop, top_crop, patch_height, patch_width)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Sparse Depth shape: {sample['depth_sparse'].shape}")
        print(f"  Ground Truth Depth shape: {sample['depth_gt'].shape}")
        print(f"  K shape: {sample['K'].shape}")
        
        # Create visualization for the first sample in the batch
        if i == 0:
            rgb = sample['rgb'][0]  
            depth_gt = sample['depth_gt'][0]  
            depth_sparse = sample['depth_sparse'][0]  
        
            rgb_np = rgb.clone()
            # Denormalize RGB (reverse ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_np = rgb_np * std + mean
            rgb_np = torch.clamp(rgb_np, 0, 1)
            rgb_np = rgb_np.permute(1, 2, 0).numpy()  
            
            depth_gt_np = depth_gt.squeeze(0).numpy()  
            depth_sparse_np = depth_sparse.squeeze(0).numpy() 
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RGB image
            axes[0].imshow(rgb_np)
            axes[0].set_title('RGB Image')
            axes[0].axis('off')
            
            # Ground truth depth
            im1 = axes[1].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=depth_gt_np.max())
            axes[1].set_title('Ground Truth Depth')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Sparse depth
            im2 = axes[2].imshow(depth_sparse_np, cmap='plasma', vmin=0, vmax=depth_gt_np.max())
            axes[2].set_title(f'Sparse Depth ({np.sum(depth_sparse_np > 0)} points)')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f'assets/kitti_sample.png', dpi=150, bbox_inches='tight')
            plt.show()
            break