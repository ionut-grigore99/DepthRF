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


# Minimize randomness
torch.manual_seed(7240)
np.random.seed(7240)
random.seed(7240)
torch.cuda.manual_seed_all(7240)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

 
def train(conf):
    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    # Prepare dataset
    data = get_data(args)

    data_train = data(args, 'train')
    data_val = data(args, 'val')
    data_test = data(args, 'test')

    sampler_train = DistributedSampler(data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(data_val, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=False, num_workers=args.num_threads, pin_memory=True, sampler=sampler_train, drop_last=True)
    loader_val = DataLoader(dataset=data_val, batch_size=1, shuffle=False, num_workers=args.num_threads, pin_memory=True, sampler=sampler_val, drop_last=False)
    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=args.num_threads)

    # Network
    model = get_model(args)
    net = model(args)
    net.cuda(gpu)

    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))

    # Loss
    loss = get_loss(args)
    loss = loss(args)
    loss.cuda(gpu)

    # Optimizer
    if args.split_backbone_training:
        optimizer, scheduler = utility.make_optimizer_scheduler_split(args, net)
    else:
        optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)

    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])

                    print('Resume optimizer, scheduler and amp from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. Use --save_full argument')

            del checkpoint

    net = DDP(net)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    if gpu == 0:
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
            os.makedirs(args.save_dir + '/test', exist_ok=True)
        except OSError:
            pass

    if gpu == 0:
        writer_train = summary(args.save_dir, 'train', args, loss.loss_name, metric.metric_name)
        writer_val = summary(args.save_dir, 'val', args, loss.loss_name, metric.metric_name)
        writer_test = summary(args.save_dir, 'test', args, loss.loss_name, metric.metric_name)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    for epoch in range(1, args.epochs+1):
        # Train
        net.train()

        sampler_train.set_epoch(epoch)

        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                print(g['lr'])
                list_lr.append(g['lr'])

            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='.format(epoch, args.epochs, list_lr, current_time, args.save_dir))

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()
            output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format('Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str, list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, output)

            if args.save_full or epoch == args.epochs:
                state = {'net': net.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'amp': amp.state_dict(), 'args': args}
            else:
                state = {'net': net.module.state_dict(), 'args': args}

            torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

        # Val
        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(loader_val) * loader_val.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}

            output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format('Val', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_val.update(epoch, sample, output)
            print('')

            writer_val.save(epoch, batch, sample, output)


        ### inline test

        num_sample = len(loader_test) * loader_test.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_test):
            sample = {key: test.cuda(gpu) for key, test in sample.items() if test is not None}

            output = net(sample)

            loss_sum, loss_test = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_test.batch_size
            loss_val = loss_test / loader_test.batch_size

            if gpu == 0:
                metric_test = metric.evaluate(sample, output, 'train')
                writer_test.add(loss_val, metric_test)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format('test', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(loader_test.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_test.update(epoch, sample, output)
            print('')

            writer_test.save(epoch, batch, sample, output)


        torch.set_grad_enabled(True)

        scheduler.step()


if __name__ == '__main__':
    lt.monkey_patch()

    conf = Config().conf

    train(conf)