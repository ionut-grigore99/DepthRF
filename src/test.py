import time
import random
import os
import json
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.conf import Config
from src.datasets.nyu import NYU
from src.datasets.kitti import KITTI
from src.models.diffusion_depth_model import DiffusionDepth  
from src.metrics import DepthRF_Metric
from src.summary import DepthRF_Summary


# Minimize randomness
torch.manual_seed(7240)
np.random.seed(7240)
random.seed(7240)
torch.cuda.manual_seed_all(7240)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def inference(conf):
    # Prepare dataset
    if conf['dataset'] == 'nyu':
        test_dataset = NYU(data_dir=conf['nyu']['data_dir'], split_json=conf['nyu']['split_json'], use_aug=False, mode='test', num_sparse_points=conf['nyu']['num_sparse_points'], height=conf['nyu']['height'], width=conf['nyu']['width'], crop_size=conf['nyu']['crop_size'])
    elif conf['dataset'] == 'kitti':
        test_dataset = KITTI(data_dir=conf['kitti']['data_dir'], split_json=conf['kitti']['split_json'], use_aug=False, mode='test', num_sparse_points=conf['kitti']['num_sparse_points'], test_crop=conf['kitti']['test_crop'], top_crop=conf['kitti']['top_crop'], patch_height=conf['kitti']['patch_height'], patch_width=conf['kitti']['patch_width'])
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=conf['num_workers'])
    
    # Get Model
    model = DiffusionDepth(conf)
    model = model.to(conf['device'])
    model.from_pretrained(weights_path=conf['model_weights'])
    model.eval()
    
    # Prepare metric and summary writer
    metric = DepthRF_Metric(conf)
    writer_test = DepthRF_Summary(log_dir='experiments', mode='test', conf=conf, loss_name=None, metric_name=metric.metric_name)

    num_sample = len(test_dataloader)*test_dataloader.batch_size
    pbar = tqdm(total=num_sample)
    time_total = 0

    for batch, sample in enumerate(test_dataloader):
        sample = {key: val.cuda() for key, val in sample.items() if val is not None}

        t0 = time.time()
        with torch.no_grad():
            output = model(sample)
        t1 = time.time()
        time_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'train')
        writer_test.add(None, metric_val)

        # Save data for analysis
        if conf['save_inference_images']:
            writer_test.save(conf['epochs'], batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        pbar.set_description(error_str)
        pbar.update(test_dataloader.batch_size)
        if batch == 0:
            break
    pbar.close()

    writer_test.update(conf['epochs'], sample, output)
    time_avg = time_total / num_sample
    print('Elapsed time : {} sec, Average processing time : {} sec'.format(time_total, time_avg))


if __name__ == '__main__':
    lt.monkey_patch()

    conf = Config().conf

    inference(conf)