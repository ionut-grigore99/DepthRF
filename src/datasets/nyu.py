"""
    NYU Depth V2 Dataset Loader.

    NYUDepthV2 json file has a following format:

{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}
"""


import os
import warnings
import numpy as np
import json
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import lovely_tensors as lt
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.utils import ToNumpy, get_sparse_depth, simple_depth_completion

warnings.filterwarnings("ignore", category=UserWarning)


class NYU(Dataset):
    def __init__(self, data_dir, split_json, use_aug, mode, num_sparse_points, height, width, crop_size):
        super(NYU, self).__init__()

        self.data_dir = data_dir # Root directory of NYUDepthV2 dataset
        self.split_json = split_json # Path to the split JSON file
        self.use_aug = use_aug # Whether to use data augmentation
        self.mode = mode # 'train', 'val', or 'test'
        self.num_sparse_points = num_sparse_points # Number of sparse depth points to sample from the dense depth map
        self.height = height # Height to resize images to
        self.width = width # Width to resize images to
        self.crop_size = crop_size # Crop size for images

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])

        with open(split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.data_dir, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        depth_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5)
        depth = Image.fromarray(depth_h5.astype('float32'))
        
        if self.use_aug and self.mode == 'train': # apply data augmentations only during training as flipping, rotation, color jittering and transformations such as resizing, cropping and normalization 
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)

            rgb = TF.rotate(rgb, angle=degree, interpolation=TF.InterpolationMode.NEAREST)
            depth = TF.rotate(depth, angle=degree, interpolation=TF.InterpolationMode.NEAREST)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet normalization values: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
            ])

            t_depth = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            depth = t_depth(depth)

            depth = depth / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
        else: # for validation and test the augmentations are not applied, only transformations such as resizing, cropping and normalization
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet normalization values: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
            ])

            t_depth = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            depth = t_depth(depth)

            K = self.K.clone()

        if self.num_sparse_points > 0:
            depth_sparse = get_sparse_depth(depth, self.num_sparse_points)
        else:
            depth_sparse = depth

        # Create binary mask for valid depth pixels
        valid_depth_mask = (depth_sparse > 0)

        # Apply simple depth completion to fill gaps in sparse depth map
        depth_filled = np.asarray(depth_sparse.squeeze(0), dtype=np.float32)  
        depth_filled, _ = simple_depth_completion(depth_filled)
        depth_filled = depth_filled[np.newaxis, ...]  

        output = {'rgb': rgb, 'depth_sparse': depth_sparse, 'depth_gt': depth, 'K': K, 'valid_depth_mask': valid_depth_mask, 'depth_filled': depth_filled}

        return output


if __name__ == "__main__":
    lt.monkey_patch() # Enable lovely_tensors for better debugging

    # Configurations
    data_dir = '/data/DepthRF/nyudepthv2'
    split_json = '/home/ionut/DepthRF/src/datasets/nyu.json'
    use_aug = True
    mode = 'train'
    num_sparse_points = 500
    height = 240
    width = 320
    crop_size = (228, 304)

    dataset = NYU(data_dir, split_json, use_aug, mode, num_sparse_points, height, width, crop_size)
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
            plt.savefig(f'assets/nyu_sample.png', dpi=150, bbox_inches='tight')
            plt.show()
            break