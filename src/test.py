import time
import random
import os
import json
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.conf import Config
from src.datasets.nyu import NYU
from src.datasets.kitti import KITTI
from src.models.diffusion_depth_model import DiffusionDepth  
from src.metrics.metrics import DepthRF_Metric

# Minimize randomness
torch.manual_seed(7240)
np.random.seed(7240)
random.seed(7240)
torch.cuda.manual_seed_all(7240)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get colormap for depth visualization
cm = plt.get_cmap('plasma') # 'plasma' or 'jet'

def save_sample_results(sample, output, save_dir, idx, conf):
    """Save visualization results for a single sample.
    """

    # ImageNet normalization constants
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    # Create sample directory
    sample_dir = os.path.join(save_dir, f'{idx:08d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        # Extract and move to CPU
        rgb = sample['rgb'].detach().clone()
        depth_sparse = sample['depth_sparse'].detach()[0, 0].cpu().numpy()
        pred = output['pred'].detach()[0, 0].cpu().numpy()
        depth_gt = sample['depth_gt'].detach()[0, 0].cpu().numpy()
        
        # Process RGB
        rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb)) # Un-normalize RGB
        rgb = rgb[0].cpu().numpy()  # [3, H, W]
        rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
        rgb = np.clip(rgb * 255.0, 0, 255).astype('uint8')
        
        # Normalize depth maps to [0, 1]
        max_depth = conf['max_depth']
        depth_sparse = np.clip(depth_sparse, 0, max_depth) / max_depth
        pred = np.clip(pred, 0, max_depth) / max_depth
        depth_gt = np.clip(depth_gt, 0, max_depth) / max_depth
        
        # Apply colormap to depth maps
        depth_sparse_colored = (255.0 * cm(depth_sparse)).astype('uint8')[:, :, :3]
        pred_colored = (255.0 * cm(pred)).astype('uint8')[:, :, :3]
        depth_gt_colored = (255.0 * cm(depth_gt)).astype('uint8')[:, :, :3]

        # Also save grayscale prediction
        pred_gray = (pred * 255.0).astype('uint8')
        
        # Convert to PIL Images
        rgb_img = Image.fromarray(rgb, 'RGB')
        depth_sparse_img = Image.fromarray(depth_sparse_colored, 'RGB')
        pred_img = Image.fromarray(pred_colored, 'RGB')
        pred_gray_img = Image.fromarray(pred_gray, 'L')
        depth_gt_img = Image.fromarray(depth_gt_colored, 'RGB')

        # Save visualization images
        rgb_img.save(os.path.join(sample_dir, '01_rgb.png'))
        depth_sparse_img.save(os.path.join(sample_dir, '02_depth_sparse.png'))
        pred_img.save(os.path.join(sample_dir, '03_pred_final.png'))
        pred_gray_img.save(os.path.join(sample_dir, '04_pred_final_gray.png'))
        depth_gt_img.save(os.path.join(sample_dir, '05_depth_gt.png'))


def inference(conf):
    # Create test directories
    test_dir = os.path.join(conf['save_dir'], 'test')
    results_dir = os.path.join(test_dir, 'visualizations')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Prepare dataset
    if conf['dataset'] == 'kitti':
        test_dataset = KITTI(data_dir=conf['kitti']['data_dir'], split_json=conf['kitti']['split_json'], use_aug=False, mode='test', test_crop=conf['kitti']['test_crop'], top_crop=conf['kitti']['top_crop'], height=conf['kitti']['height'], width=conf['kitti']['width'])
    elif conf['dataset'] == 'nyu':
        test_dataset = NYU(data_dir=conf['nyu']['data_dir'], split_json=conf['nyu']['split_json'], use_aug=False, mode='test', num_sparse_points=conf['nyu']['num_sparse_points'], height=conf['nyu']['height'], width=conf['nyu']['width'], center_crop_size=conf['nyu']['center_crop_size'])
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=conf['num_workers'])

    # Get Model
    model = DiffusionDepth(conf)
    model = model.to(conf['device'])
    model.from_pretrained(weights_path=conf['model_weights'])
    model.eval()
    
    # Prepare metric
    metric = DepthRF_Metric()

    all_metrics = {name: [] for name in metric.metric_name}
    num_sample = len(test_dataloader)*test_dataloader.batch_size
    time_total = 0
    print(f"\nTesting on {num_sample} samples...")
    print(f"Results will be saved to: {test_dir}\n")

    for batch, sample in enumerate(tqdm(test_dataloader, desc='Testing')):
        sample = {key: val.cuda() for key, val in sample.items() if val is not None}
   
        t0 = time.time()
        with torch.no_grad():
            output = model(sample)
        t1 = time.time()
        time_total += (t1 - t0)

        metrics = metric.evaluate(sample, output)
        metrics = metrics.squeeze().cpu().numpy()
        for name, value in zip(metric.metric_name, metrics):
            all_metrics[name].append(value)

        # Save data for analysis
        if conf['save_test_images']:
            save_sample_results(sample, output, results_dir, batch, conf)

        # if batch == 0:
        #     break # For debugging, process only first batch

    avg_metrics = {name: np.mean(values) for name, values in all_metrics.items()}
    time_avg = time_total / num_sample
    fps = 1.0 / time_avg if time_avg > 0 else 0

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Dataset: {conf['dataset']}")
    print(f"Model: {conf['model_weights']}")
    print(f"Samples: {num_sample}")
    print(f"Total time: {time_total:.2f} sec")
    print(f"Average time: {time_avg:.4f} sec/sample")
    print(f"FPS: {fps:.2f}")
    print("-"*60)
    print("Metrics:")
    for name, value in avg_metrics.items():
        print(f"  {name:<15s}: {value:.4f}")
    print("="*60)

    metrics_file = os.path.join(test_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: {conf['dataset']}\n")
        f.write(f"Model: {conf['model_weights']}\n")
        f.write(f"Samples: {num_sample}\n")
        f.write(f"Total time: {time_total:.2f} sec\n")
        f.write(f"Average time: {time_avg:.4f} sec/sample\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write("-"*60 + "\n")
        f.write("Metrics:\n")
        for name, value in avg_metrics.items():
            f.write(f"  {name:<15s}: {value:.4f}\n")
        f.write("="*60 + "\n")

if __name__ == '__main__':
    lt.monkey_patch()

    conf = Config().conf
    conf['save_test_images'] = False # Ensure test images are saved

    inference(conf)