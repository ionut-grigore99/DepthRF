from importlib import import_module
import torch
import torch.nn as nn

def l1_loss(pred, gt, max_depth=80.0, t_valid=0.0001):
    """L1 loss for depth prediction."""
    gt = torch.clamp(gt, min=0, max=max_depth)
    pred = torch.clamp(pred, min=0, max=max_depth)

    mask = (gt > t_valid).type_as(pred).detach()
    d = torch.abs(pred - gt) * mask

    d = torch.sum(d, dim=[1, 2, 3])
    num_valid = torch.sum(mask, dim=[1, 2, 3])

    loss = d / (num_valid + 1e-8)
    return loss.sum()

def l2_loss(pred, gt, max_depth=80.0, t_valid=0.0001):
    """L2 loss for depth prediction."""
    gt = torch.clamp(gt, min=0, max=max_depth)
    pred = torch.clamp(pred, min=0, max=max_depth)

    mask = (gt > t_valid).type_as(pred).detach()
    d = torch.pow(pred - gt, 2) * mask

    d = torch.sum(d, dim=[1, 2, 3])
    num_valid = torch.sum(mask, dim=[1, 2, 3])

    loss = d / (num_valid + 1e-8)
    return loss.sum()

def compute_depth_loss(sample, output, conf):
    """Compute depth loss based on configuration."""

    pred = output['pred']
    gt = sample['depth_gt']
    
    loss_dict = {}
    total_loss = 0.0

    # Parse loss configuration: "1.0*L1+1.0*L2+1.0*DDIM" for example
    loss_config = conf['loss']
    
    for loss_item in loss_config.split('+'):
        weight_str, loss_type = loss_item.split('*')
        weight = float(weight_str)
        loss_type = loss_type.strip().upper()
        
        if loss_type == 'L1':
            loss_value = l1_loss(pred, gt, max_depth=conf['max_depth'])
        
        elif loss_type == 'L2':
            loss_value = l2_loss(pred, gt, max_depth=conf['max_depth'])

        elif loss_type == 'DDIM':
            loss_value = output['ddim_loss']

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Weight and accumulate
        weighted_loss = weight * loss_value
        loss_dict[loss_type] = weighted_loss.item()
        total_loss = total_loss + weighted_loss
    
    loss_dict['Total'] = total_loss.item()

    return total_loss, loss_dict
