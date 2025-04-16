from __future__ import print_function

import random

import torch
import torch.nn.functional as F


def ensure_size(input_tensor, min_size=40):
    """Ensure tensor is at least min_size x min_size by padding if necessary."""
    _, _, h, w = input_tensor.shape
    if h >= min_size and w >= min_size:
        return input_tensor
    
    # Calculate padding
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    
    # Pad tensor symmetrically
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    return F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')


def aug(input_tensor):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size
    
    # Ensure all images are at least 40x40 (to support 32x32 crops)
    input_tensor = ensure_size(input_tensor, min_size=40)
    
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)
    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        # Adjust max random offset based on actual image size
        max_x = max(0, input_tensor.shape[2] - 32)
        max_y = max(0, input_tensor.shape[3] - 32)
        x_t = random.randint(0, max_x)
        y_t = random.randint(0, max_y)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def aug_trans(input_tensor, transform_info):
    batch_size = input_tensor.shape[0]
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flip = transform_info['flipped']
    
    # Ensure all images are at least 40x40 (to support 32x32 crops)
    input_tensor = ensure_size(input_tensor, min_size=40)
    
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = int(flip[i])
        x_t = int(x[i])
        y_t = int(y[i])
        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
    return rst


def inverse_aug(source_tensor, adv_tensor, transform_info):
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flipped = transform_info['flipped']
    batch_size = source_tensor.shape[0]

    # Ensure source tensor is large enough
    source_tensor = ensure_size(source_tensor, min_size=40)

    for i in range(batch_size):
        flip_t = int(flipped[i])
        x_t = int(x[i])
        y_t = int(y[i])
        if flip_t:
            adv_tensor[i] = torch.flip(adv_tensor[i], [2])
        source_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32] = adv_tensor[i]

    return source_tensor


def aug_imagenet(input_tensor):
    input_tensor = F.interpolate(input_tensor, (256, 256), mode='bilinear')
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size
    rst = torch.zeros((len(input_tensor), 3, 224, 224), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 32)
        y_t = random.randint(0, 32)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def aug_trans_imagenet(input_tensor, transform_info):
    batch_size = input_tensor.shape[0]
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flip = transform_info['flipped']
    rst = torch.zeros((len(input_tensor), 3, 224, 224), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = int(flip[i])
        x_t = int(x[i])
        y_t = int(y[i])
        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
    return rst


def inverse_aug_imagenet(source_tensor, adv_tensor, transform_info):
    interpolate_tensor = F.interpolate(source_tensor, (256, 256), mode='bilinear')
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flipped = transform_info['flipped']
    batch_size = source_tensor.shape[0]

    for i in range(batch_size):
        flip_t = int(flipped[i])
        x_t = int(x[i])
        y_t = int(y[i])
        if flip_t:
            adv_tensor[i] = torch.flip(adv_tensor[i], [2])
        interpolate_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224] = adv_tensor[i]

    return F.interpolate(source_tensor, (32, 32), mode='bilinear')
