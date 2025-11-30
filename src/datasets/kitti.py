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

from src.utils import get_sparse_depth

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
    def __init__(self, data_dir, split_json, use_aug, mode, test_crop, top_crop, height, width):
        super(KITTI, self).__init__()

        self.data_dir = data_dir # Root directory of KITTI dataset
        self.split_json = split_json # Path to the split JSON file
        self.use_aug = use_aug # Whether to use data augmentation
        self.mode = mode # 'train', 'val', or 'test'
        self.test_crop = test_crop # Whether to apply cropping during testing
        self.top_crop = top_crop # Number of pixels to crop from the top of the image
        self.height = height # Height of the cropped patch
        self.width = width # Width of the cropped patch
        

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        with open(self.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth_velodyne, depth_accumulated = self._load_data(idx)

        if self.use_aug and self.mode == 'train': # apply data augmentations only during training as flipping, rotation, color jittering and transformations such as resizing, cropping and normalization 
            # Top crop if needed
            if self.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth_velodyne = TF.crop(depth_velodyne, self.top_crop, 0, height - self.top_crop, width)
                depth_accumulated = TF.crop(depth_accumulated, self.top_crop, 0, height - self.top_crop, width)

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth_velodyne = TF.hflip(depth_velodyne)
                depth_accumulated = TF.hflip(depth_accumulated)

            # Rotation
            rgb = TF.rotate(rgb, angle=degree, resample=Image.BICUBIC)
            depth_velodyne = TF.rotate(depth_velodyne, angle=degree, resample=Image.NEAREST)
            depth_accumulated = TF.rotate(depth_accumulated, angle=degree, resample=Image.NEAREST)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            # Resize
            rgb = TF.resize(rgb, scale, Image.BICUBIC)
            depth_velodyne = TF.resize(depth_velodyne, scale, Image.NEAREST)
            depth_accumulated = TF.resize(depth_accumulated, scale, Image.NEAREST)

            # Crop
            width, height = rgb.size
          
            assert self.height <= height and self.width <= width, "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth_velodyne = TF.crop(depth_velodyne, h_start, w_start, self.height, self.width)
            depth_accumulated = TF.crop(depth_accumulated, h_start, w_start, self.height, self.width)

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True) # ImageNet normalization values: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

            depth_velodyne = TF.to_tensor(np.array(depth_velodyne))
            depth_velodyne = depth_velodyne / _scale

            depth_accumulated = TF.to_tensor(np.array(depth_accumulated))
            depth_accumulated = depth_accumulated / _scale
        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth_velodyne = TF.crop(depth_velodyne, self.top_crop, 0, height - self.top_crop, width)
                depth_accumulated = TF.crop(depth_accumulated, self.top_crop, 0, height - self.top_crop, width)

            # Crop
            width, height = rgb.size
            
            assert self.height <= height and self.width <= width, "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth_velodyne = TF.crop(depth_velodyne, h_start, w_start, self.height, self.width)
            depth_accumulated = TF.crop(depth_accumulated, h_start, w_start, self.height, self.width)

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True) # ImageNet normalization values: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

            depth_velodyne = TF.to_tensor(np.array(depth_velodyne))

            depth_accumulated = TF.to_tensor(np.array(depth_accumulated))
        else:
            if self.top_crop > 0 and self.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.top_crop, 0, height - self.top_crop, width)
                depth_velodyne = TF.crop(depth_velodyne, self.top_crop, 0, height - self.top_crop, width)
                gt = TF.crop(gt, self.top_crop, 0, height - self.top_crop, width)

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True) # ImageNet normalization values: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

            depth_velodyne = TF.to_tensor(np.array(depth_velodyne))

            depth_accumulated = TF.to_tensor(np.array(depth_accumulated))


        output = {'rgb': rgb, 'depth_sparse': depth_velodyne, 'depth_gt': depth_accumulated}

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.data_dir, self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.data_dir, self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.data_dir, self.sample_list[idx]['gt'])

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt


if __name__ == "__main__":
    lt.monkey_patch()  # Enable lovely_tensors for better debugging

    # Configurations
    data_dir = '/data/DepthRF/kitti/kitti_dp'
    split_json = '/home/ionut/DepthRF/src/datasets/kitti.json'
    use_aug = True
    mode = 'train'
    test_crop = False
    height = 240
    width = 1216
    top_crop = 100

    dataset = KITTI(data_dir, split_json, use_aug, mode, test_crop, top_crop, height, width)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Sparse Depth shape: {sample['depth_sparse'].shape}")
        print(f"  Ground Truth Depth shape: {sample['depth_gt'].shape}")
        
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
            axes[2].set_title('Sparse Depth')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f'assets/kitti_sample.png', dpi=150, bbox_inches='tight')
            plt.show()
            break