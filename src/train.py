import time
import random
import os
import json
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.conf import Config
from src.models.diffusion_depth_model import DiffusionDepth  
from src.datasets.nyu import NYU
from src.datasets.kitti import KITTI
from src.metrics.metrics import DepthRF_Metric
from src.losses.loss import compute_depth_loss
from src.utils import backup_source_code


# Minimize randomness
torch.manual_seed(7240)
np.random.seed(7240)
random.seed(7240)
torch.cuda.manual_seed_all(7240)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get colormap for depth visualization
cm = plt.get_cmap('plasma') # 'plasma' or 'jet'

def visualize_batch_to_tensorboard(writer, sample, output, global_step, mode, conf):
    """Visualize a batch of predictions in TensorBoard."""

    # ImageNet normalization constants
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    
    with torch.no_grad():
        # Un-normalize RGB
        rgb = sample['rgb'].detach().clone()
        rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))
        rgb = rgb.cpu().numpy()
        
        # Get depth maps
        depth_sparse = sample['depth_sparse'].detach().cpu().numpy()
        depth_gt = sample['depth_gt'].detach().cpu().numpy()
        pred = output['pred'].detach().cpu().numpy()
        
        # Limit number of images to visualize
        num_summary = min(rgb.shape[0], conf['num_summary'])
        
        rgb = rgb[0:num_summary, :, :, :]
        depth_sparse = depth_sparse[0:num_summary, :, :, :]
        depth_gt = depth_gt[0:num_summary, :, :, :]
        pred = pred[0:num_summary, :, :, :]
        
        # Clip values
        max_depth = conf['max_depth']
        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        depth_sparse = np.clip(depth_sparse, a_min=0, a_max=max_depth)
        depth_gt = np.clip(depth_gt, a_min=0, a_max=max_depth)
        pred = np.clip(pred, a_min=0, a_max=max_depth)
        
        list_img = []
        
        for b in range(num_summary):
            # Extract single sample [C, H, W] or [H, W]
            rgb_tmp = rgb[b, :, :, :]  # [3, H, W]
            depth_sparse_tmp = depth_sparse[b, 0, :, :]   # [H, W]
            depth_gt_tmp = depth_gt[b, 0, :, :]     # [H, W]
            pred_tmp = pred[b, 0, :, :] # [H, W]
            
            # Normalize depth to [0, 255] for colormap
            depth_sparse_tmp = 255.0 * depth_sparse_tmp / max_depth
            depth_gt_tmp = 255.0 * depth_gt_tmp / max_depth
            pred_tmp = 255.0 * pred_tmp / max_depth
            
            # Apply colormap (returns RGBA, we take RGB only)
            depth_sparse_tmp = cm(depth_sparse_tmp.astype('uint8'))[:, :, :3]    # [H, W, 3]
            depth_gt_tmp = cm(depth_gt_tmp.astype('uint8'))[:, :, :3]      # [H, W, 3]
            pred_tmp = cm(pred_tmp.astype('uint8'))[:, :, :3]  # [H, W, 3]

            depth_sparse_tmp = np.transpose(depth_sparse_tmp, (2, 0, 1))
            depth_gt_tmp = np.transpose(depth_gt_tmp, (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp, (2, 0, 1))

            # Concatenate vertically: RGB | Sparse Depth | Prediction | Ground Truth Depth  
            img = np.concatenate((rgb_tmp, depth_sparse_tmp, pred_tmp, depth_gt_tmp), axis=1)

            list_img.append(img)
        
        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)
        
        writer.add_image(f'{mode}/Visualization', img_total, global_step)
 
def train(conf):
    # Prepare dataset
    if conf['dataset'] == 'kitti':
        train_dataset = KITTI(conf['kitti']['data_dir'], conf['kitti']['split_json'], use_aug=True, mode='train', num_sparse_points=conf['kitti']['num_sparse_points'], test_crop=False, top_crop=conf['kitti']['top_crop'], patch_height=conf['kitti']['patch_height'], patch_width=conf['kitti']['patch_width'])
        val_dataset = KITTI(conf['kitti']['data_dir'], conf['kitti']['split_json'], use_aug=False, mode='val', num_sparse_points=conf['kitti']['num_sparse_points'], test_crop=False, top_crop=conf['kitti']['top_crop'], patch_height=conf['kitti']['patch_height'], patch_width=conf['kitti']['patch_width'])
        test_dataset = KITTI(conf['kitti']['data_dir'], conf['kitti']['split_json'], use_aug=False, mode='test', num_sparse_points=0, test_crop=conf['kitti']['test_crop'], top_crop=conf['kitti']['top_crop'], patch_height=conf['kitti']['patch_height'], patch_width=conf['kitti']['patch_width'])
    elif conf['dataset'] == 'nyu':
        train_dataset = NYU(conf['nyu']['data_dir'], conf['nyu']['split_json'], use_aug=True, mode='train', num_sparse_points=conf['nyu']['num_sparse_points'])
        val_dataset = NYU(conf['nyu']['data_dir'], conf['nyu']['split_json'], use_aug=False, mode='val', num_sparse_points=conf['nyu']['num_sparse_points'])
        test_dataset = NYU(conf['nyu']['data_dir'], conf['nyu']['split_json'], use_aug=False, mode='test', num_sparse_points=0)
    else:
        raise ValueError(f"Unknown dataset: {conf['dataset']}")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=conf['batch_size'], shuffle=True, num_workers=conf['num_workers'], pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=conf['num_workers'], pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=conf['num_workers'])
    
    # Get Model
    model = DiffusionDepth(conf)
    model = model.to(conf['device'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], betas=conf['betas'], eps=conf['epsilon'], weight_decay=conf['weight_decay'])
    
    # Scheduler
    lr_decay = list(map(int, conf["lr_decay"].split(",")))
    lr_gamma = list(map(float, conf["lr_gamma"].split(",")))
    assert len(lr_decay) == len(lr_gamma), 'lr_decay and lr_gamma must have same length'
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: next((g for d, g in zip(lr_decay, lr_gamma) if epoch < d), lr_gamma[-1]))

    # Resume from checkpoint if specified
    if conf['resume_checkpoint'] is not None:
        checkpoint = torch.load(conf['resume_checkpoint'])
        key_m, key_u = self.load_state_dict(checkpoint['net'], strict=True)
        if key_u:
            print('Unexpected keys :')
            print(key_u)
        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Resume optimizer and scheduler from : {}'.format(conf['resume_checkpoint']))
        except KeyError:
            print('State dicts for resume are not saved. Use --save_full argument')
        del checkpoint

    # Prepare metrics and loss names
    metrics = DepthRF_Metric()
    loss_names = []
    for loss_item in conf['loss'].split('+'):
        _, loss_type = loss_item.split('*')
        loss_names.append(loss_type.strip().upper())
    loss_names.append('Total')
    
    # Prepare logging and TensorBoard ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path(conf['save_dir']) / 'train' / timestamp
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "config.yaml", "w") as config_yaml:
        yaml.dump(conf, config_yaml, default_flow_style=False, sort_keys=False)

    # Save source code as a sanity check
    backup_source_code(logdir / 'backup_code')
    writer = SummaryWriter(log_dir=logdir)
    with open(logdir / "config.yaml", "r") as config_yaml:
        config_text = config_yaml.read()
        writer.add_text("config", f"```yaml\n{config_text}\n```", global_step=0)

    # Prepare warm-up for learning rate if specified
    if conf['warm_up']:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(train_dataloader)+1.0

    # Get one fixed batch from each dataset for consistent tensorboard visualization across epochs
    train_vis_sample, val_vis_sample, test_vis_sample = [
        {k: v.cuda() for k, v in next(iter(loader)).items() if v is not None}
        for loader in [train_dataloader, val_dataloader, test_dataloader]
    ]

    for epoch in range(1, conf['epochs']+1):
        ##############################
        #          TRAINING          #
        ##############################
        model.train()
        epoch_loss_sum = {k: 0.0 for k in loss_names}
        epoch_metrics_sum = {name: 0.0 for name in metrics.metric_name}
        for batch, sample in enumerate(tqdm(train_dataloader, desc='Training')):
            sample = {key: val.cuda() for key, val in sample.items() if val is not None}
            
            if epoch == 1 and conf['warm_up']:
                warm_up_cnt += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
            
            optimizer.zero_grad()
            output = model(sample)
            train_loss, train_loss_dict = compute_depth_loss(sample, output, conf)
      
            # Divide all losses by batch size
            train_loss = train_loss / train_dataloader.batch_size
            train_loss_dict = {k: v / train_dataloader.batch_size for k, v in train_loss_dict.items()}

            train_loss.backward()
            optimizer.step()

            metrics_train = metrics.evaluate(sample, output).squeeze().cpu().numpy()
            
            # Accumulate losses and metrics for epoch average
            for key, value in train_loss_dict.items():
                epoch_loss_sum[key] += value
            
            for name, value in zip(metrics.metric_name, metrics_train):
                epoch_metrics_sum[name] += value
            
        # Log epoch averages to TensorBoard
        num_train_batches = len(train_dataloader)
        for loss_name, loss_sum in epoch_loss_sum.items():
            writer.add_scalar(f'Train/Loss/{loss_name}', loss_sum / num_train_batches, epoch)
        
        for metric_name, metric_sum in epoch_metrics_sum.items():
            writer.add_scalar(f'Train/Metrics/{metric_name}', metric_sum / num_train_batches, epoch)
      
        # Log current learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Visualize training samples
        with torch.no_grad():
            train_vis_output = model(train_vis_sample)
        visualize_batch_to_tensorboard(writer, train_vis_sample, train_vis_output, epoch, 'Train', conf)

        if conf['save_full_model'] or epoch == conf['epochs']:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'conf': conf}
        else:
            state = {'net': model.state_dict(), 'conf': conf}
        torch.save(state, '{}/model_{:05d}.pt'.format(logdir, epoch))

        ##############################
        #          VALIDATION        #
        ##############################
        torch.set_grad_enabled(False)
        model.eval()
        val_loss_sum = {k: 0.0 for k in loss_names}
        val_metrics_sum = {name: 0.0 for name in metrics.metric_name}
        for batch, sample in enumerate(tqdm(val_dataloader, desc='Validation')):
            sample = {key: val.cuda() for key, val in sample.items() if val is not None}
            output = model(sample)
            val_loss, val_loss_dict = compute_depth_loss(sample, output, conf)

            # Divide by batch size
            val_loss = val_loss / val_dataloader.batch_size
            val_loss_dict = {k: v / val_dataloader.batch_size for k, v in val_loss_dict.items()}

            metrics_val = metrics.evaluate(sample, output).squeeze().cpu().numpy()

            # Accumulate losses and metrics for epoch average
            for key, value in val_loss_dict.items():
                val_loss_sum[key] += value
            
            for name, value in zip(metrics.metric_name, metrics_val):
                val_metrics_sum[name] += value

        # Log epoch averages to TensorBoard
        num_val_batches = len(val_dataloader)
        for loss_name, loss_sum in val_loss_sum.items():
            writer.add_scalar(f'Val/Loss/{loss_name}', loss_sum / num_val_batches, epoch)

        for metric_name, metric_sum in val_metrics_sum.items():
            writer.add_scalar(f'Val/Metrics/{metric_name}', metric_sum / num_val_batches, epoch)

        # Visualize validation samples
        val_vis_output = model(val_vis_sample)
        visualize_batch_to_tensorboard(writer, val_vis_sample, val_vis_output, epoch, 'Val', conf)

        ##############################
        #          TESTING           #
        ##############################
        test_loss_sum = {k: 0.0 for k in loss_names}
        test_metrics_sum = {name: 0.0 for name in metrics.metric_name}
        for batch, sample in enumerate(tqdm(test_dataloader, desc='Testing')):
            sample = {key: test.cuda() for key, test in sample.items() if test is not None}
            output = model(sample)
            test_loss, test_loss_dict = compute_depth_loss(sample, output, conf)

            # Divide by batch size
            test_loss = test_loss / test_dataloader.batch_size
            test_loss_dict = {k: v / test_dataloader.batch_size for k, v in test_loss_dict.items()}

            metrics_test = metrics.evaluate(sample, output).squeeze().cpu().numpy()
            
            # Accumulate losses and metrics for epoch average
            for key, value in test_loss_dict.items():
                test_loss_sum[key] += value

            for name, value in zip(metrics.metric_name, metrics_test):
                test_metrics_sum[name] += value

        # Log epoch averages to TensorBoard
        num_test_batches = len(test_dataloader)
        for loss_name, loss_sum in test_loss_sum.items():
            writer.add_scalar(f'Test/Loss/{loss_name}', loss_sum / num_test_batches, epoch)

        for metric_name, metric_sum in test_metrics_sum.items():
            writer.add_scalar(f'Test/Metrics/{metric_name}', metric_sum / num_test_batches, epoch)

        # Visualize test samples
        test_vis_output = model(test_vis_sample)
        visualize_batch_to_tensorboard(writer, test_vis_sample, test_vis_output, epoch, 'Test', conf)

        torch.set_grad_enabled(True)
        scheduler.step()

    print('Training completed. Model and logs are saved in {}'.format(logdir))

if __name__ == '__main__':
    lt.monkey_patch()

    conf = Config().conf

    train(conf)