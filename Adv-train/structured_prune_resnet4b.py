#!/usr/bin/env python3
# Structured Model Pruning Script for ResNet4b with ATAS fine-tuning
# Usage python structured_prune_resnet4b.py --model-dir ./results/cifar_atas_resnet4b_ultrawide --output-dir ./results/cifar_atas_resnet4b_ultrapruned
import argparse
import copy
import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import psutil
from collections import OrderedDict

# Custom imports from ATAS
sys.path.append('.')
from models.resnet4b import CResNet7, BasicBlock, resnet4b, resnet4b_wide, resnet4b_ultrawide
from ATAS import test_adversarial
import adv_attack
from models.normalize import Normalize

# CIFAR10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

def parse_args():
    parser = argparse.ArgumentParser(description='Structured Pruning with Adversarial Fine-tuning')
    
    # Basic parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--model-dir', default='./results/cifar_atas_resnet4b_wide', type=str, help='Directory of the trained model')
    parser.add_argument('--output-dir', default='./results/cifar_atas_resnet4b_pruned', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    # Training parameters
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--test-batch-size', default=128, type=int, help='Batch size for testing')
    parser.add_argument('--finetune-epochs', default=15, type=int, help='Number of epochs for fine-tuning')
    parser.add_argument('--clean-epochs', default=5, type=int, help='Number of epochs for clean fine-tuning after adversarial training')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for fine-tuning')
    parser.add_argument('--clean-lr', default=0.005, type=float, help='Learning rate for clean fine-tuning')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loading workers')
    
    # Pruning parameters
    parser.add_argument('--prune-rate', default=0.5, type=float, help='Target ratio for pruning (from 32 to 16 planes)')
    parser.add_argument('--prune-iterations', default=10, type=int, help='Number of pruning iterations')
    
    # Adversarial training parameters
    parser.add_argument('--epsilon', default=4.0/255, type=float, help='Perturbation size for training')
    parser.add_argument('--step-size', default=1.0/255, type=float, help='Step size for PGD')
    parser.add_argument('--pgd-steps', default=10, type=int, help='Number of PGD steps for training')
    parser.add_argument('--eval-epsilons', default='1.0,2.0,4.0', type=str, help='Comma-separated list of epsilon values for evaluation (in 1/255 units)')
    
    args = parser.parse_args()
    return args


def memory_usage():
    """Get the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def setup_logging(args):
    """Set up logging directory and files"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_path = os.path.join(args.output_dir, 'pruning_log.json')
    log_data = {
        'args': vars(args),
        'pruning_history': [],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    return log_path, log_data


def get_dataset(args):
    """Set up data loaders for CIFAR-10"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    return train_loader, test_loader, train_size, test_size


def load_model(args, device):
    """Load the pretrained model and handle potential key mismatches"""
    # Define normalization parameters
    mean = CIFAR10_MEAN
    std = CIFAR10_STD
    normalize = Normalize(mean, std)
    
    # Determine which model type to use based on model_dir name
    if 'ultrawide' in args.model_dir:
        print("Loading ultrawide model (in_planes=64)...")
        base_model = resnet4b_ultrawide()
    elif 'wide' in args.model_dir:
        print("Loading wide model (in_planes=32)...")
        base_model = resnet4b_wide()
    else:
        print("Loading standard model (in_planes=16)...")
        base_model = resnet4b()
        
    model = nn.Sequential(normalize, base_model)
    
    checkpoint_path = os.path.join(args.model_dir, 'best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print(f"=> Loading pretrained model from {checkpoint_path}")
    source_state_dict = torch.load(checkpoint_path, map_location=device)

    # --- Key remapping for the base model component --- 
    new_state_dict = OrderedDict()
    has_prefix = False
    for key in source_state_dict.keys():
        if key.startswith('1.'):
            has_prefix = True
            break
            
    if has_prefix:
        print("Detected '1.' prefix in checkpoint keys. Remapping...")
        for key, value in source_state_dict.items():
            if key.startswith('1.'):
                # The base_model is now at index 1 in Sequential
                new_key = key # Keep the '1.' prefix since our model structure uses it
                new_state_dict[new_key] = value
            else:
                # If somehow a key doesn't have the prefix, add it
                new_key = '1.' + key
                new_state_dict[new_key] = value
                print(f"  Warning: Key {key} does not have expected '1.' prefix. Added prefix.")
    else:
        print("No '1.' prefix detected. Adding '1.' prefix to match our Sequential structure...")
        for key, value in source_state_dict.items():
            new_key = '1.' + key # Add '1.' prefix to match Sequential(normalize, model)
            new_state_dict[new_key] = value
    # --- End key remapping logic ---

    # Load the state dictionary
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    
    print("Model successfully loaded with normalization layer.")
    return model


def analyze_layer_importance(model, data_loader, device, criterion=nn.CrossEntropyLoss()):
    """Analyze the importance of each filter in convolutional layers using sensitivity analysis"""
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    # Dictionary to store layer importances
    importances = {}
    
    # Get baseline performance
    model.eval()
    baseline_loss = 0
    n_samples = 0
    
    # Use a few batches for faster computation
    max_batches = 5
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            if batch_count >= max_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            baseline_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)
            batch_count += 1
    
    baseline_loss /= n_samples
    
    # Reset batch counter for importance analysis
    batch_count = 0
    
    # Analyze convolutional layers
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Analyzing importance of filters in {name}...")
            
            # Skip if it's the first layer (we want to keep all input channels)
            if name == 'conv1':
                continue
                
            # Get the number of filters in this layer
            num_filters = module.weight.size(0)
            filter_importances = []
            
            # For each filter, temporarily zero it out and measure performance drop
            for filter_idx in range(num_filters):
                # Save original weights
                original_weight = module.weight.clone()
                
                # Zero out the filter
                module.weight.data[filter_idx] = 0
                
                # Measure performance drop
                loss_with_filter_removed = 0
                n_samples = 0
                batch_count = 0
                
                with torch.no_grad():
                    for inputs, targets in data_loader:
                        if batch_count >= max_batches:
                            break
                            
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        loss_with_filter_removed += loss.item() * inputs.size(0)
                        n_samples += inputs.size(0)
                        batch_count += 1
                
                loss_with_filter_removed /= n_samples
                
                # Compute importance as the increase in loss
                importance = loss_with_filter_removed - baseline_loss
                filter_importances.append((filter_idx, importance))
                
                # Restore original weights
                module.weight.data = original_weight
            
            # Sort filters by importance (ascending, least important first)
            filter_importances.sort(key=lambda x: x[1])
            importances[name] = filter_importances
    
    return importances


def select_channels_to_prune(importances, prune_rate):
    """Select channels to prune based on importance scores"""
    prune_indices = {}
    
    for layer_name, filter_importances in importances.items():
        # Calculate how many filters to prune in this layer
        num_filters = len(filter_importances)
        num_to_prune = int(num_filters * prune_rate)
        
        # Select the least important filters
        filters_to_prune = [idx for idx, _ in filter_importances[:num_to_prune]]
        prune_indices[layer_name] = filters_to_prune
        
        print(f"Layer {layer_name}: pruning {num_to_prune}/{num_filters} filters")
    
    return prune_indices


def create_pruned_model(model, prune_indices, device, target_width='auto'):
    """Create a new, smaller model with pruned channels, properly handling all layers and dependencies
    
    Args:
        model: The model to prune
        prune_indices: Dictionary of layer name to list of filter indices to prune
        device: The device to create the new model on
        target_width: Target model width ('auto', '16', '32') where 'auto' selects based on source model
    """
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    # Determine the source model type
    if hasattr(base_model, 'in_planes'):
        source_in_planes = base_model.in_planes
    else:
        # Infer from the first conv layer if attribute isn't available
        source_in_planes = base_model.state_dict()['conv1.weight'].size(0)
    
    print(f"Source model has in_planes={source_in_planes}")
    
    # Determine target model width based on source and target_width parameter
    if target_width == 'auto':
        # Automatically determine target width based on source
        if source_in_planes == 64:  # ultrawide -> wide
            target_in_planes = 32
            target_model = resnet4b_wide()
            print("Automatically selected target: resnet4b_wide (in_planes=32)")
        elif source_in_planes == 32:  # wide -> standard
            target_in_planes = 16
            target_model = resnet4b()
            print("Automatically selected target: resnet4b (in_planes=16)")
        else:  # already at standard or smaller
            target_in_planes = 16
            target_model = resnet4b()
            print("Automatically selected target: resnet4b (in_planes=16)")
    elif target_width == '32':
        target_in_planes = 32
        target_model = resnet4b_wide()
        print("Using specified target: resnet4b_wide (in_planes=32)")
    else:  # default to smallest model (in_planes=16)
        target_in_planes = 16
        target_model = resnet4b()
        print("Using specified target: resnet4b (in_planes=16)")
    
    target_model = target_model.to(device)
    
    # Calculate pruning ratio based on source and target width
    prune_ratio = 1.0 - (float(target_in_planes) / source_in_planes)
    print(f"Pruning ratio: {prune_ratio:.2f} ({source_in_planes} -> {target_in_planes} in_planes)")
    
    # Get the state dictionaries
    source_state_dict = base_model.state_dict()
    target_state_dict = OrderedDict()
    
    # Get the target shapes for each layer from the target model
    target_shapes = {k: v.shape for k, v in target_model.state_dict().items()}
    
    # We need to track indices of kept filters for each layer to ensure consistency
    kept_indices = {}
    
    # First, process conv1 (input channels are always RGB (3))
    conv1_name = 'conv1.weight'
    if 'conv1' in prune_indices:
        filters_to_prune = prune_indices['conv1']
        all_indices = list(range(source_state_dict[conv1_name].size(0)))
        output_channels_to_keep = [i for i in all_indices if i not in filters_to_prune]
    else:
        # If not specified, take the first channels by default
        output_channels_to_keep = list(range(min(target_in_planes, source_state_dict[conv1_name].size(0))))
    
    # Ensure we have exactly target_in_planes channels
    if len(output_channels_to_keep) > target_in_planes:
        output_channels_to_keep = output_channels_to_keep[:target_in_planes]
    elif len(output_channels_to_keep) < target_in_planes:
        # If we don't have enough, add more from the full list
        all_indices = list(range(source_state_dict[conv1_name].size(0)))
        remaining = [i for i in all_indices if i not in output_channels_to_keep]
        output_channels_to_keep.extend(remaining[:target_in_planes-len(output_channels_to_keep)])
    
    # Apply pruning to conv1
    target_state_dict[conv1_name] = source_state_dict[conv1_name][output_channels_to_keep]
    kept_indices['conv1'] = output_channels_to_keep
    
    # Handle bias if present
    if 'conv1.bias' in source_state_dict:
        target_state_dict['conv1.bias'] = source_state_dict['conv1.bias'][output_channels_to_keep]
    
    # BatchNorm1 (if present)
    if 'bn1.weight' in source_state_dict:
        target_state_dict['bn1.weight'] = source_state_dict['bn1.weight'][output_channels_to_keep]
        target_state_dict['bn1.bias'] = source_state_dict['bn1.bias'][output_channels_to_keep]
        target_state_dict['bn1.running_mean'] = source_state_dict['bn1.running_mean'][output_channels_to_keep]
        target_state_dict['bn1.running_var'] = source_state_dict['bn1.running_var'][output_channels_to_keep]
    
    # The input channels for the next layer come from conv1's output channels
    prev_output_channels = output_channels_to_keep
    
    # Process BasicBlocks in layer1 and layer2
    for layer_idx in range(1, 3):  # layer1 and layer2
        layer_name = f'layer{layer_idx}'
        for block_idx in range(2):  # Each layer has 2 blocks
            block_prefix = f'{layer_name}.{block_idx}'
            
            # Get expected output channels for this block from target model
            # In both resnet4b and resnet4b_wide, layer1 and layer2 have in_planes*2 channels
            target_output_channels = target_in_planes * 2
            
            # Process first conv in block
            conv1_name = f'{block_prefix}.conv1.weight'
            if f'{block_prefix}.conv1' in prune_indices:
                filters_to_prune = prune_indices[f'{block_prefix}.conv1']
                all_indices = list(range(source_state_dict[conv1_name].size(0)))
                output_channels = [i for i in all_indices if i not in filters_to_prune]
            else:
                # If not specified, take first channels by default
                output_channels = list(range(min(target_output_channels, source_state_dict[conv1_name].size(0))))
            
            # Enforce exact number of channels for target model
            if len(output_channels) > target_output_channels:
                output_channels = output_channels[:target_output_channels]
            elif len(output_channels) < target_output_channels:
                all_indices = list(range(source_state_dict[conv1_name].size(0)))
                remaining = [i for i in all_indices if i not in output_channels]
                output_channels.extend(remaining[:target_output_channels-len(output_channels)])
            
            # Get the pruned weight tensor with correct input and output channels
            source_weight = source_state_dict[conv1_name]
            
            # Check dimensions before slicing to avoid IndexError
            if len(prev_output_channels) > source_weight.size(1):
                prev_output_channels = prev_output_channels[:source_weight.size(1)]
            
            # Apply pruning to conv1 of this block
            # Handles case where we need different output and input channel counts
            pruned_weight = torch.zeros(
                target_shapes[conv1_name], 
                device=source_weight.device,
                dtype=source_weight.dtype
            )
            
            # Copy the selected channels' weights
            # Only copy what fits in both source and target
            out_channels = min(len(output_channels), target_shapes[conv1_name][0])
            in_channels = min(len(prev_output_channels), target_shapes[conv1_name][1])
            
            for i in range(out_channels):
                for j in range(in_channels):
                    if i < len(output_channels) and j < len(prev_output_channels):
                        pruned_weight[i, j] = source_weight[output_channels[i], prev_output_channels[j]]
            
            target_state_dict[conv1_name] = pruned_weight
            kept_indices[f'{block_prefix}.conv1'] = output_channels
            
            # Handle bias if present
            if f'{block_prefix}.conv1.bias' in source_state_dict:
                bias = source_state_dict[f'{block_prefix}.conv1.bias']
                pruned_bias = torch.zeros(
                    target_shapes[f'{block_prefix}.conv1.bias'],
                    device=bias.device,
                    dtype=bias.dtype
                )
                out_channels = min(len(output_channels), target_shapes[f'{block_prefix}.conv1.bias'][0])
                for i in range(out_channels):
                    if i < len(output_channels):
                        pruned_bias[i] = bias[output_channels[i]]
                target_state_dict[f'{block_prefix}.conv1.bias'] = pruned_bias
            
            # BatchNorm after first conv
            bn1_name = f'{block_prefix}.bn1'
            if f'{bn1_name}.weight' in source_state_dict:
                for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    full_name = f'{bn1_name}.{param_name}'
                    param = source_state_dict[full_name]
                    pruned_param = torch.zeros(
                        target_shapes[full_name],
                        device=param.device,
                        dtype=param.dtype
                    )
                    out_channels = min(len(output_channels), pruned_param.size(0))
                    for i in range(out_channels):
                        if i < len(output_channels):
                            pruned_param[i] = param[output_channels[i]]
                    target_state_dict[full_name] = pruned_param
            
            # Process second conv in block
            conv2_name = f'{block_prefix}.conv2.weight'
            # For conv2, output channels should be the same as input
            # for the residual connection to work
            
            # Get the pruned weight tensor 
            source_weight = source_state_dict[conv2_name]
            pruned_weight = torch.zeros(
                target_shapes[conv2_name],
                device=source_weight.device,
                dtype=source_weight.dtype
            )
            
            # Copy the selected channels' weights
            for i in range(min(target_shapes[conv2_name][0], len(output_channels))):
                for j in range(min(target_shapes[conv2_name][1], len(output_channels))):
                    if i < len(output_channels) and j < len(output_channels):
                        pruned_weight[i, j] = source_weight[output_channels[i], output_channels[j]]
            
            target_state_dict[conv2_name] = pruned_weight
            kept_indices[f'{block_prefix}.conv2'] = output_channels
            
            # Handle bias if present
            if f'{block_prefix}.conv2.bias' in source_state_dict:
                bias = source_state_dict[f'{block_prefix}.conv2.bias']
                pruned_bias = torch.zeros(
                    target_shapes[f'{block_prefix}.conv2.bias'],
                    device=bias.device,
                    dtype=bias.dtype
                )
                out_channels = min(len(output_channels), target_shapes[f'{block_prefix}.conv2.bias'][0])
                for i in range(out_channels):
                    if i < len(output_channels):
                        pruned_bias[i] = bias[output_channels[i]]
                target_state_dict[f'{block_prefix}.conv2.bias'] = pruned_bias
            
            # BatchNorm after second conv
            bn2_name = f'{block_prefix}.bn2'
            if f'{bn2_name}.weight' in source_state_dict:
                for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    full_name = f'{bn2_name}.{param_name}'
                    param = source_state_dict[full_name]
                    pruned_param = torch.zeros(
                        target_shapes[full_name],
                        device=param.device,
                        dtype=param.dtype
                    )
                    out_channels = min(len(output_channels), pruned_param.size(0))
                    for i in range(out_channels):
                        if i < len(output_channels):
                            pruned_param[i] = param[output_channels[i]]
                    target_state_dict[full_name] = pruned_param
            
            # Handle shortcut if it exists
            shortcut_name = f'{block_prefix}.shortcut.0.weight'
            if shortcut_name in source_state_dict and shortcut_name in target_shapes:
                source_weight = source_state_dict[shortcut_name]
                pruned_weight = torch.zeros(
                    target_shapes[shortcut_name],
                    device=source_weight.device,
                    dtype=source_weight.dtype
                )
                
                # Copy the selected channels' weights
                for i in range(min(target_shapes[shortcut_name][0], len(output_channels))):
                    for j in range(min(target_shapes[shortcut_name][1], len(prev_output_channels))):
                        if i < len(output_channels) and j < len(prev_output_channels):
                            pruned_weight[i, j] = source_weight[output_channels[i], prev_output_channels[j]]
                
                target_state_dict[shortcut_name] = pruned_weight
                
                # Handle bias if present
                shortcut_bias_name = f'{block_prefix}.shortcut.0.bias'
                if shortcut_bias_name in source_state_dict:
                    bias = source_state_dict[shortcut_bias_name]
                    pruned_bias = torch.zeros(
                        target_shapes[shortcut_bias_name],
                        device=bias.device,
                        dtype=bias.dtype
                    )
                    out_channels = min(len(output_channels), target_shapes[shortcut_bias_name][0])
                    for i in range(out_channels):
                        if i < len(output_channels):
                            pruned_bias[i] = bias[output_channels[i]]
                    target_state_dict[shortcut_bias_name] = pruned_bias
                
                # BatchNorm in shortcut if present
                bn_shortcut_name = f'{block_prefix}.shortcut.1'
                if f'{bn_shortcut_name}.weight' in source_state_dict:
                    for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                        full_name = f'{bn_shortcut_name}.{param_name}'
                        param = source_state_dict[full_name]
                        pruned_param = torch.zeros(
                            target_shapes[full_name],
                            device=param.device,
                            dtype=param.dtype
                        )
                        out_channels = min(len(output_channels), pruned_param.size(0))
                        for i in range(out_channels):
                            if i < len(output_channels):
                                pruned_param[i] = param[output_channels[i]]
                        target_state_dict[full_name] = pruned_param
            
            # Update prev_output_channels for the next block
            prev_output_channels = output_channels
    
    # Handle linear layers
    if 'linear1.weight' in source_state_dict:
        # Linear1 needs special handling due to the flattened features
        # The target shape is fixed for the target model
        target_weight = target_model.state_dict()['linear1.weight']
        target_state_dict['linear1.weight'] = target_weight.clone()  # Use the initialized weights
        target_state_dict['linear1.bias'] = source_state_dict['linear1.bias']  # Bias is the same
        
        # Linear2 remains unchanged
        target_state_dict['linear2.weight'] = source_state_dict['linear2.weight']
        target_state_dict['linear2.bias'] = source_state_dict['linear2.bias']
    
    # Log what we've done
    print("\nStructured Pruning Summary:")
    print(f"Original model: in_planes={source_in_planes}")
    print(f"Pruned model: in_planes={target_in_planes}")
    for layer_name, indices in kept_indices.items():
        print(f"Layer {layer_name}: Kept {len(indices)} channels")
    
    # Apply the pruned state dict to the new model
    try:
        target_model.load_state_dict(target_state_dict)
        print("\nSuccessfully created pruned model with matching architecture!")
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        # Print detailed shape comparison to debug
        print("\nDetailed shape comparison:")
        target_shapes = {k: v.shape for k, v in target_model.state_dict().items()}
        for k in target_shapes:
            if k in target_state_dict:
                if target_shapes[k] != target_state_dict[k].shape:
                    print(f"  Mismatch in {k}: Target={target_shapes[k]}, Pruned={target_state_dict[k].shape}")
    
    # Create a new full model with normalization
    mean = CIFAR10_MEAN
    std = CIFAR10_STD
    normalize = Normalize(mean, std)
    pruned_model = nn.Sequential(normalize, target_model).to(device)
    
    return pruned_model


def pgd_attack(model, images, labels, epsilon, step_size, steps, device):
    """
    Perform PGD attack on the given images
    """
    # Make a copy of the original images
    perturbed_images = images.clone().detach()
    
    # Initialize with random noise within epsilon ball
    perturbed_images = perturbed_images + torch.zeros_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    for step in range(steps):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        
        # Get gradient
        grad = torch.autograd.grad(loss, perturbed_images, retain_graph=False, create_graph=False)[0]
        
        # Update perturbed images
        perturbed_images = perturbed_images.detach() + step_size * grad.sign()
        
        # Project back to epsilon ball and valid image range
        delta = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = torch.clamp(images + delta, 0, 1)
    
    return perturbed_images.detach()


def train_epoch(model, train_loader, optimizer, device, args):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    epsilon = 4.0/255  # Fixed to 4/255 for adversarial training
    step_size = args.step_size
    steps = args.pgd_steps
    
    print(f"Training with adversarial examples: epsilon={epsilon*255:.1f}/255, steps={steps}")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples using PGD
        adv_inputs = pgd_attack(model, inputs, targets, epsilon=epsilon, step_size=step_size, steps=steps, device=device)
        
        # Forward pass and loss calculation with adversarial examples
        optimizer.zero_grad()
        outputs = model(adv_inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.2f}%")
    
    return train_loss / len(train_loader), 100. * correct / total


def train_clean_epoch(model, train_loader, optimizer, device):
    """Train for one epoch on clean examples only to recover natural accuracy"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with clean examples
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Clean Batch: {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {train_loss/total:.4f}, "
                  f"Acc: {100.*correct/total:.2f}%, "
                  f"Memory: {memory_usage():.1f} MB")
    
    train_time = time.time() - start_time
    train_loss = train_loss / total
    train_acc = 100. * correct / total
    
    print(f"Clean Train Time: {train_time:.2f}s, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    return train_loss, train_acc


def evaluate(model, data_loader, device, epsilon=None, steps=10):
    """Evaluate the model on the test set, with or without adversarial examples"""
    model.eval()
    
    if epsilon is None:
        # Clean evaluation
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / total
        test_acc = 100. * correct / total
    else:
        # Adversarial evaluation
        test_loss = 0
        correct = 0
        total = 0
        step_size = epsilon / 4  # Smaller step size for evaluation
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize with random perturbation
            x_adv = inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # PGD attack
            for _ in range(steps):
                x_adv.requires_grad_(True)
                outputs = model(x_adv)
                loss = F.cross_entropy(outputs, targets)
                
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                
                # Project back into epsilon ball and valid image space
                delta = torch.clamp(x_adv - inputs, -epsilon, epsilon)
                x_adv = torch.clamp(inputs + delta, 0, 1)
            
            # Final evaluation
            with torch.no_grad():
                outputs = model(x_adv)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / total
        test_acc = 100. * correct / total
    
    return test_loss, test_acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Adjust pruning rate based on model type
    if 'ultrawide' in args.model_dir:
        if args.prune_rate < 0.7:
            original_rate = args.prune_rate
            args.prune_rate = 0.75
            print(f"Adjusting pruning rate from {original_rate} to {args.prune_rate} for ultrawide model (64→16 in_planes)")
    elif 'wide' in args.model_dir:
        if args.prune_rate < 0.4:
            original_rate = args.prune_rate
            args.prune_rate = 0.5
            print(f"Adjusting pruning rate from {original_rate} to {args.prune_rate} for wide model (32→16 in_planes)")
    
    # Set up logging
    log_path, log_data = setup_logging(args)
    
    # Get data loaders
    train_loader, test_loader, train_size, test_size = get_dataset(args)
    print(f"Train set: {train_size} samples, Test set: {test_size} samples")
    
    # Load the pretrained model
    model = load_model(args, device)
    
    # Determine source model width from model name
    if 'ultrawide' in args.model_dir:
        source_width = 64
        print("Source model is ultrawide (in_planes=64)")
        # We'll use staged pruning: ultrawide->wide->standard
        stages = ['32', '16']
        stage_names = ['wide', 'standard']
    elif 'wide' in args.model_dir:
        source_width = 32
        print("Source model is wide (in_planes=32)")
        # Only need one stage: wide->standard
        stages = ['16']
        stage_names = ['standard']
    else:
        source_width = 16
        print("Source model is standard (in_planes=16)")
        # No pruning needed, but we'll still run iterations for fine-tuning
        stages = ['16']
        stage_names = ['standard']
    
    # Evaluate the original model
    print("\n=== Evaluating original model ===")
    orig_natural_loss, orig_natural_acc = evaluate(model, test_loader, device)
    
    # Evaluate on different epsilon values
    eval_epsilons = [float(eps)/255 for eps in args.eval_epsilons.split(',')]
    orig_adv_accs = {}
    
    for eps in eval_epsilons:
        print(f"Evaluating with epsilon={eps*255:.1f}/255")
        # PGD-10 evaluation
        _, adv_acc_10 = evaluate(model, test_loader, device, epsilon=eps, steps=10)
        # PGD-50 evaluation
        _, adv_acc_50 = evaluate(model, test_loader, device, epsilon=eps, steps=50)
        
        orig_adv_accs[f"eps_{eps*255:.1f}_pgd10"] = adv_acc_10
        orig_adv_accs[f"eps_{eps*255:.1f}_pgd50"] = adv_acc_50
        
        print(f"Original model - PGD-10 @ {eps*255:.1f}/255: {adv_acc_10:.2f}%")
        print(f"Original model - PGD-50 @ {eps*255:.1f}/255: {adv_acc_50:.2f}%")
    
    print(f"Original model - Natural Accuracy: {orig_natural_acc:.2f}%")
    
    # Log original model results
    log_data['original_model'] = {
        'natural_acc': orig_natural_acc,
        'adversarial_accs': orig_adv_accs,
    }
    
    # Initialize pruning
    current_model = model
    
    # Process each stage sequentially
    for stage_idx, (target_width, stage_name) in enumerate(zip(stages, stage_names)):
        print(f"\n\n=== Starting STAGE {stage_idx+1}/{len(stages)}: Pruning to {stage_name} (in_planes={target_width}) ===\n")
        
        # Calculate how many iterations for this stage
        if len(stages) == 1:
            # If only one stage, use all iterations
            stage_iterations = args.prune_iterations
        else:
            # Distribute iterations proportionally to the pruning ratio
            if stage_idx == 0:  # First stage (64->32)
                stage_iterations = args.prune_iterations // 2
            else:  # Second stage (32->16)
                stage_iterations = args.prune_iterations - (args.prune_iterations // 2)
        
        print(f"Running {stage_iterations} pruning iterations for this stage")
        
        # Reset the current model's prune ratio for this stage
        if target_width == '32':
            stage_prune_ratio = 1.0 - (32.0 / source_width)
        else:  # target_width == '16'
            if source_width == 64 and stage_idx == 1:  # Second stage from wide(32) to standard(16)
                stage_prune_ratio = 0.5  # 32 to 16 is 50% pruning
            else:
                stage_prune_ratio = 1.0 - (16.0 / source_width)
                
        print(f"Stage pruning ratio: {stage_prune_ratio:.2f}")
        
        # Calculate per-iteration pruning rate
        iteration_prune_rate = stage_prune_ratio / stage_iterations
        
        # Iterate through pruning steps for this stage
        for iteration in range(1, stage_iterations + 1):
            print(f"\n=== Stage {stage_idx+1} Pruning Iteration {iteration}/{stage_iterations} ===")
            
            # Analyze importance for current model
            importances = analyze_layer_importance(current_model, train_loader, device)
            
            # Select channels to prune
            prune_indices = select_channels_to_prune(importances, iteration_prune_rate)
            
            # Create a new pruned model with the target width for this stage
            pruned_model = create_pruned_model(
                current_model, 
                prune_indices, 
                device, 
                target_width=target_width
            )
            
            # Evaluate the pruned model before fine-tuning
            print("\n=== Evaluating pruned model (before fine-tuning) ===")
            pruned_nat_loss, pruned_nat_acc = evaluate(pruned_model, test_loader, device)
            print(f"Pruned model - Natural Accuracy: {pruned_nat_acc:.2f}% (before fine-tuning)")
            
            # Fine-tune the pruned model
            print(f"\n=== Fine-tuning for {args.finetune_epochs} epochs ===")
            
            optimizer = optim.SGD(
                pruned_model.parameters(),
                lr=args.lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
            
            best_adv_acc_4 = 0  # Track best adversarial accuracy at 4/255
            best_model_state = None
            
            for epoch in range(1, args.finetune_epochs + 1):
                print(f"\nEpoch {epoch}/{args.finetune_epochs}")
                
                # Train for one epoch with adversarial examples
                train_loss, train_acc = train_epoch(pruned_model, train_loader, optimizer, device, args)
                
                # Evaluate
                val_nat_loss, val_nat_acc = evaluate(pruned_model, test_loader, device)
                
                # Evaluate on epsilon=1/255 for tracking progress (reporting purposes)
                eval_eps_1 = eval_epsilons[0]  # First epsilon (1/255)
                _, val_adv_acc_1 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_1, steps=10)
                
                # Evaluate on epsilon=4/255 for model selection
                eval_eps_4 = 4.0/255  # Use 4/255 directly
                _, val_adv_acc_4 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_4, steps=10)
                
                print(f"Epoch {epoch} - Natural: {val_nat_acc:.2f}%, PGD-10 @ 1/255: {val_adv_acc_1:.2f}%, PGD-10 @ 4/255: {val_adv_acc_4:.2f}%")
                
                # Update learning rate
                scheduler.step()
                
                # Save best model based on 4/255 adversarial accuracy
                if val_adv_acc_4 > best_adv_acc_4:
                    best_adv_acc_4 = val_adv_acc_4
                    best_model_state = copy.deepcopy(pruned_model.state_dict())
                    print(f"New best model with adversarial accuracy {best_adv_acc_4:.2f}% at epsilon=4/255")
            
            # Restore best model
            if best_model_state is not None:
                pruned_model.load_state_dict(best_model_state)
                
            # Additional clean fine-tuning to recover natural accuracy
            if args.clean_epochs > 0:
                print(f"\n=== Clean fine-tuning for {args.clean_epochs} epochs to recover natural accuracy ===")
                
                clean_optimizer = optim.SGD(
                    pruned_model.parameters(),
                    lr=args.clean_lr, 
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                )
                clean_scheduler = optim.lr_scheduler.CosineAnnealingLR(clean_optimizer, T_max=args.clean_epochs)
                
                best_nat_acc = 0
                best_nat_model_state = None
                
                for epoch in range(1, args.clean_epochs + 1):
                    print(f"\nClean Epoch {epoch}/{args.clean_epochs}")
                    
                    # Train for one epoch with clean examples
                    train_loss, train_acc = train_clean_epoch(pruned_model, train_loader, clean_optimizer, device)
                    
                    # Evaluate
                    val_nat_loss, val_nat_acc = evaluate(pruned_model, test_loader, device)
                    
                    # Evaluate on epsilon=1/255 (reporting purposes)
                    eval_eps_1 = eval_epsilons[0]  
                    _, val_adv_acc_1 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_1, steps=10)
                    
                    # Evaluate on epsilon=4/255 (for model selection)
                    eval_eps_4 = 4.0/255
                    _, val_adv_acc_4 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_4, steps=10)
                    
                    print(f"Clean Epoch {epoch} - Natural: {val_nat_acc:.2f}%, PGD-10 @ 1/255: {val_adv_acc_1:.2f}%, PGD-10 @ 4/255: {val_adv_acc_4:.2f}%")
                    
                    # Update learning rate
                    clean_scheduler.step()
                    
                    # Save best model based on natural accuracy, with a constraint on adversarial accuracy
                    # Only save if natural accuracy improves and adversarial accuracy at 4/255 doesn't drop too much
                    if val_nat_acc > best_nat_acc and val_adv_acc_4 >= best_adv_acc_4 * 0.95:  # Allow 5% drop in adversarial accuracy
                        best_nat_acc = val_nat_acc
                        best_nat_model_state = copy.deepcopy(pruned_model.state_dict())
                        print(f"New best clean model with natural accuracy {best_nat_acc:.2f}% and adversarial accuracy {val_adv_acc_4:.2f}% at epsilon=4/255")
                
                # Restore best clean model if it improves natural accuracy without compromising robustness
                if best_nat_model_state is not None:
                    pruned_model.load_state_dict(best_nat_model_state)
                    print(f"Restored best clean model with natural accuracy {best_nat_acc:.2f}%")
            
            # If this is the last iteration of the stage, evaluate on all epsilons
            if iteration == stage_iterations:
                print(f"\n=== Evaluating stage {stage_idx+1} model ===")
                final_nat_loss, final_nat_acc = evaluate(pruned_model, test_loader, device)
                final_adv_accs = {}
                
                for eps in eval_epsilons:
                    print(f"Evaluating with epsilon={eps*255:.1f}/255")
                    # PGD-10 evaluation
                    _, adv_acc_10 = evaluate(pruned_model, test_loader, device, epsilon=eps, steps=10)
                    # PGD-50 evaluation
                    _, adv_acc_50 = evaluate(pruned_model, test_loader, device, epsilon=eps, steps=50)
                    
                    final_adv_accs[f"eps_{eps*255:.1f}_pgd10"] = adv_acc_10
                    final_adv_accs[f"eps_{eps*255:.1f}_pgd50"] = adv_acc_50
                    
                    print(f"Stage {stage_idx+1} model - PGD-10 @ {eps*255:.1f}/255: {adv_acc_10:.2f}%")
                    print(f"Stage {stage_idx+1} model - PGD-50 @ {eps*255:.1f}/255: {adv_acc_50:.2f}%")
                
                print(f"Stage {stage_idx+1} model - Natural Accuracy: {final_nat_acc:.2f}%")
                
                # Log results for this stage
                log_data[f'stage_{stage_idx+1}'] = {
                    'target_width': target_width,
                    'stage_name': stage_name,
                    'natural_acc': final_nat_acc,
                    'adversarial_accs': final_adv_accs,
                }
                
                # Save to log file
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Save the fine-tuned model for this stage
                base_model = pruned_model[1] if isinstance(pruned_model, nn.Sequential) else pruned_model
                torch.save(base_model.state_dict(), 
                          os.path.join(args.output_dir, f'pruned_stage_{stage_idx+1}_{stage_name}.pth'))
            else:
                # Log results for this iteration including both 1/255 and 4/255 metrics
                log_data['pruning_history'].append({
                    'stage': stage_idx+1,
                    'iteration': iteration,
                    'natural_acc': val_nat_acc,
                    'adversarial_acc_eps1': val_adv_acc_1,
                    'adversarial_acc_eps4': val_adv_acc_4,
                })
                
                # Save to log file
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Save the intermediate model
                base_model = pruned_model[1] if isinstance(pruned_model, nn.Sequential) else pruned_model
                torch.save(base_model.state_dict(), 
                          os.path.join(args.output_dir, f'pruned_stage_{stage_idx+1}_iter_{iteration}.pth'))
            
            # Update current model for next iteration
            current_model = pruned_model
    
    # Save the final pruned model
    final_model = current_model[1] if isinstance(current_model, nn.Sequential) else current_model
    torch.save(final_model.state_dict(), os.path.join(args.output_dir, 'final_pruned.pth'))
    
    print("\n=== Structured pruning completed successfully ===")
    print(f"Results saved to {args.output_dir}")
    print(f"Log file: {log_path}")


if __name__ == '__main__':
    main() 
 