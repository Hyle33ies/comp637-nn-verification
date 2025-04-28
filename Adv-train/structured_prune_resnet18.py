#!/usr/bin/env python3
# Structured Model Pruning Script for ResNet18 with ATAS fine-tuning
# Usage python structured_prune_resnet18.py --model-dir ./results/cifar_atas_resnet18_2 --output-dir ./results/cifar_atas_resnet18_pruned
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

# Custom imports
sys.path.append('.')
from models.resnet import BasicBlock, ResNet18, ResNet18_32, ResNet18_16, ResNet18_8, ResNet18_4
from ATAS import test_adversarial # Assuming this is still relevant
import adv_attack # Assuming this is still relevant
from models.normalize import Normalize

# CIFAR10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

def parse_args():
    parser = argparse.ArgumentParser(description='Structured Pruning for ResNet18 with Adversarial Fine-tuning')
    
    # Basic parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--model-dir', default='./results/cifar_atas_resnet18', type=str, help='Directory of the trained model')
    parser.add_argument('--output-dir', default='./results/cifar_atas_resnet18_pruned', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    # Training parameters
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--test-batch-size', default=128, type=int, help='Batch size for testing')
    parser.add_argument('--finetune-epochs', default=20, type=int, help='Default number of epochs for adversarial fine-tuning per iteration')
    parser.add_argument('--clean-epochs', default=0, type=int, help='Default number of epochs for clean fine-tuning per iteration')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for fine-tuning')
    parser.add_argument('--clean-lr', default=0.005, type=float, help='Learning rate for clean fine-tuning')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loading workers')
    
    # Pruning parameters
    parser.add_argument('--prune-rate', default=0.875, type=float, help='Overall target ratio for pruning (from 64 to 8 planes)')
    parser.add_argument('--prune-iterations', default=8, type=int, help='Total number of pruning iterations across all stages (ignored if stage-specific iterations are set)')
    parser.add_argument('--stage1-iterations', default=2, type=int, help='Number of iterations for Stage 1 (64->32)')
    parser.add_argument('--stage2-iterations', default=2, type=int, help='Number of iterations for Stage 2 (32->16)')
    parser.add_argument('--stage3-iterations', default=4, type=int, help='Number of iterations for Stage 3 (16->4)')
    parser.add_argument('--stage1-finetune-epochs', default=3, type=int, help='Adversarial epochs per iteration in Stage 1')
    parser.add_argument('--stage2-finetune-epochs', default=3, type=int, help='Adversarial epochs per iteration in Stage 2')
    parser.add_argument('--stage3-finetune-epochs', default=15, type=int, help='Adversarial epochs per iteration in Stage 3 (16->4)')
    parser.add_argument('--stage1-clean-epochs', default=1, type=int, help='Clean epochs per iteration in Stage 1')
    parser.add_argument('--stage2-clean-epochs', default=1, type=int, help='Clean epochs per iteration in Stage 2')
    parser.add_argument('--stage3-clean-epochs', default=0, type=int, help='Clean epochs per iteration in Stage 3 (16->4)')
    
    # Adversarial training parameters
    parser.add_argument('--epsilon', default=4.0/255, type=float, help='Perturbation size for training')
    parser.add_argument('--step-size', default=1.0/255, type=float, help='Step size for PGD')
    parser.add_argument('--pgd-steps', default=10, type=int, help='Number of PGD steps for training')
    parser.add_argument('--eval-epsilons', default='1.0,2.0,4.0,8.0', type=str, help='Comma-separated list of epsilon values for evaluation (in 1/255 units)')
    
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
    
    # Assuming the model directory indicates the original model type (ResNet18 default)
    print("Loading standard ResNet18 model (in_planes=64)...")
    base_model = ResNet18()
    
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
        target_width: Target model width ('auto', '32', '16', '8') where 'auto' selects next stage
    """
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    # Determine the source model type (in_planes)
    if hasattr(base_model, 'in_planes'):
        # This attribute is tricky in ResNet as it changes during _make_layer
        # So, infer from conv1 output channels instead
        source_in_planes = base_model.state_dict()['conv1.weight'].size(0)
    else:
        source_in_planes = base_model.state_dict()['conv1.weight'].size(0)
    
    print(f"Source model has initial in_planes={source_in_planes}")
    
    # Determine target model width based on source and target_width parameter
    if target_width == 'auto':
        if source_in_planes == 64: target_in_planes = 32
        elif source_in_planes == 32: target_in_planes = 16
        elif source_in_planes == 16: target_in_planes = 4 # Changed from 8 to 4
        else: target_in_planes = 4 # Default to smallest
    else:
        target_in_planes = int(target_width)
        
    # Select the correct target model function
    if target_in_planes == 32: target_model_func = ResNet18_32
    elif target_in_planes == 16: target_model_func = ResNet18_16
    elif target_in_planes == 8: target_model_func = ResNet18_8 # Keep this for potential use
    elif target_in_planes == 4: target_model_func = ResNet18_4 # Added ResNet18_4
    else: 
        print(f"Warning: Unsupported target_in_planes {target_in_planes}. Defaulting to ResNet18_4.")
        target_in_planes = 4
        target_model_func = ResNet18_4
        
    target_model = target_model_func()
    print(f"Targeting model with in_planes={target_in_planes}")
    
    target_model = target_model.to(device)
    
    # Calculate pruning ratio based on source and target width
    # Ensure source_in_planes is not zero
    if source_in_planes == 0: 
        raise ValueError("Source model in_planes cannot be zero.")
    prune_ratio = 1.0 - (float(target_in_planes) / source_in_planes)
    print(f"Pruning ratio for this stage: {prune_ratio:.2f} ({source_in_planes} -> {target_in_planes} in_planes)")
    
    # Get the state dictionaries
    source_state_dict = base_model.state_dict()
    target_state_dict = OrderedDict()
    
    # Get the target shapes for each layer from the target model
    target_shapes = {k: v.shape for k, v in target_model.state_dict().items()}
    
    # We need to track indices of kept filters for each layer to ensure consistency
    kept_indices = {}
    
    # First, process conv1 and bn1 (input channels are always RGB (3))
    conv1_name = 'conv1.weight'
    if 'conv1' in prune_indices: # Usually we don't prune conv1, but handle if specified
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
        all_indices = list(range(source_state_dict[conv1_name].size(0)))
        remaining = [i for i in all_indices if i not in output_channels_to_keep]
        output_channels_to_keep.extend(remaining[:target_in_planes-len(output_channels_to_keep)])
    
    # Apply pruning to conv1
    target_state_dict[conv1_name] = source_state_dict[conv1_name][output_channels_to_keep]
    kept_indices['conv1'] = output_channels_to_keep
    
    # Handle bias if present (ResNet18 uses bias=False for conv1)
    if 'conv1.bias' in source_state_dict and 'conv1.bias' in target_shapes:
        target_state_dict['conv1.bias'] = source_state_dict['conv1.bias'][output_channels_to_keep]
    
    # BatchNorm1
    for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
        bn1_param_name = f'bn1.{param_name}'
        if bn1_param_name in source_state_dict:
            target_state_dict[bn1_param_name] = source_state_dict[bn1_param_name][output_channels_to_keep]
    
    # The input channels for layer1 come from conv1's output channels
    prev_output_channels = output_channels_to_keep
    
    # Process BasicBlocks in layer1, layer2, layer3, layer4
    # Correct plane calculation for ResNet18 structure
    layer_planes_map = { 32: [32, 64, 128, 256], 
                         16: [16, 32, 64, 128],
                         8:  [8, 16, 32, 64],
                         4:  [4, 8, 16, 32] }
    layer_planes = layer_planes_map.get(target_in_planes, layer_planes_map[4]) # Default to 4 if invalid

    for layer_idx in range(1, 5):  # layer1 to layer4
        layer_name = f'layer{layer_idx}'
        target_layer_planes = layer_planes[layer_idx-1]
        
        # Each layer in ResNet18 has 2 blocks
        for block_idx in range(2):  
            block_prefix = f'{layer_name}.{block_idx}'
            
            # --- Process first conv in block --- 
            conv1_name = f'{block_prefix}.conv1.weight'
            source_conv1_weight = source_state_dict[conv1_name]
            target_conv1_shape = target_shapes[conv1_name]
            target_conv1_output_channels = target_conv1_shape[0]
            
            if f'{block_prefix}.conv1' in prune_indices:
                filters_to_prune = prune_indices[f'{block_prefix}.conv1']
                all_indices = list(range(source_conv1_weight.size(0)))
                output_channels = [i for i in all_indices if i not in filters_to_prune]
            else:
                # Default: Keep first channels up to target output size
                output_channels = list(range(min(target_conv1_output_channels, source_conv1_weight.size(0))))
            
            # Enforce exact number of output channels for target model
            if len(output_channels) > target_conv1_output_channels:
                output_channels = output_channels[:target_conv1_output_channels]
            elif len(output_channels) < target_conv1_output_channels:
                all_indices = list(range(source_conv1_weight.size(0)))
                remaining = [i for i in all_indices if i not in output_channels]
                output_channels.extend(remaining[:target_conv1_output_channels-len(output_channels)])
            
            # Apply pruning to conv1 weight
            # Input channels: prev_output_channels
            # Output channels: output_channels
            pruned_weight = torch.zeros(target_conv1_shape, device=source_conv1_weight.device, dtype=source_conv1_weight.dtype)
            out_ch_copy = min(len(output_channels), target_conv1_shape[0])
            in_ch_copy = min(len(prev_output_channels), target_conv1_shape[1])
            
            for i in range(out_ch_copy):
                for j in range(in_ch_copy):
                    if i < len(output_channels) and j < len(prev_output_channels):
                        pruned_weight[i, j] = source_conv1_weight[output_channels[i], prev_output_channels[j]]
            
            target_state_dict[conv1_name] = pruned_weight
            kept_indices[f'{block_prefix}.conv1'] = output_channels
            
            # Handle conv1 bias if present
            conv1_bias_name = f'{block_prefix}.conv1.bias'
            if conv1_bias_name in source_state_dict and conv1_bias_name in target_shapes:
                bias = source_state_dict[conv1_bias_name]
                pruned_bias = torch.zeros(target_shapes[conv1_bias_name], device=bias.device, dtype=bias.dtype)
                out_ch_copy = min(len(output_channels), target_shapes[conv1_bias_name][0])
                for i in range(out_ch_copy):
                    if i < len(output_channels):
                        pruned_bias[i] = bias[output_channels[i]]
                target_state_dict[conv1_bias_name] = pruned_bias
            
            # BatchNorm after first conv (bn1)
            bn1_name = f'{block_prefix}.bn1'
            for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                bn1_param_name = f'{bn1_name}.{param_name}'
                if bn1_param_name in source_state_dict:
                    param = source_state_dict[bn1_param_name]
                    pruned_param = torch.zeros(target_shapes[bn1_param_name], device=param.device, dtype=param.dtype)
                    out_ch_copy = min(len(output_channels), pruned_param.size(0))
                    for i in range(out_ch_copy):
                        if i < len(output_channels):
                            pruned_param[i] = param[output_channels[i]]
                    target_state_dict[bn1_param_name] = pruned_param
            
            # --- Process second conv in block --- 
            conv2_name = f'{block_prefix}.conv2.weight'
            source_conv2_weight = source_state_dict[conv2_name]
            target_conv2_shape = target_shapes[conv2_name]
            # For conv2, output channels should be the same as input channels (which are output_channels from conv1)
            
            # Apply pruning to conv2 weight
            pruned_weight = torch.zeros(target_conv2_shape, device=source_conv2_weight.device, dtype=source_conv2_weight.dtype)
            out_ch_copy = min(len(output_channels), target_conv2_shape[0])
            in_ch_copy = min(len(output_channels), target_conv2_shape[1])
            
            for i in range(out_ch_copy):
                for j in range(in_ch_copy):
                    if i < len(output_channels) and j < len(output_channels):
                        pruned_weight[i, j] = source_conv2_weight[output_channels[i], output_channels[j]]
            
            target_state_dict[conv2_name] = pruned_weight
            kept_indices[f'{block_prefix}.conv2'] = output_channels
            
            # Handle conv2 bias if present
            conv2_bias_name = f'{block_prefix}.conv2.bias'
            if conv2_bias_name in source_state_dict and conv2_bias_name in target_shapes:
                bias = source_state_dict[conv2_bias_name]
                pruned_bias = torch.zeros(target_shapes[conv2_bias_name], device=bias.device, dtype=bias.dtype)
                out_ch_copy = min(len(output_channels), target_shapes[conv2_bias_name][0])
                for i in range(out_ch_copy):
                    if i < len(output_channels):
                        pruned_bias[i] = bias[output_channels[i]]
                target_state_dict[conv2_bias_name] = pruned_bias
            
            # BatchNorm after second conv (bn2)
            bn2_name = f'{block_prefix}.bn2'
            for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                bn2_param_name = f'{bn2_name}.{param_name}'
                if bn2_param_name in source_state_dict and bn2_param_name in target_shapes:
                    param = source_state_dict[bn2_param_name]
                    pruned_param = torch.zeros(target_shapes[bn2_param_name], device=param.device, dtype=param.dtype)
                    out_ch_copy = min(len(output_channels), pruned_param.size(0))
                    for i in range(out_ch_copy):
                        if i < len(output_channels):
                            pruned_param[i] = param[output_channels[i]]
                    target_state_dict[bn2_param_name] = pruned_param
            
            # --- Handle shortcut connection --- 
            shortcut_conv_name = f'{block_prefix}.shortcut.0.weight'
            if shortcut_conv_name in source_state_dict and shortcut_conv_name in target_shapes:
                source_shortcut_weight = source_state_dict[shortcut_conv_name]
                target_shortcut_shape = target_shapes[shortcut_conv_name]
                
                # Shortcut input comes from prev_output_channels, output goes to output_channels
                pruned_weight = torch.zeros(target_shortcut_shape, device=source_shortcut_weight.device, dtype=source_shortcut_weight.dtype)
                out_ch_copy = min(len(output_channels), target_shortcut_shape[0])
                in_ch_copy = min(len(prev_output_channels), target_shortcut_shape[1])
                
                for i in range(out_ch_copy):
                    for j in range(in_ch_copy):
                        if i < len(output_channels) and j < len(prev_output_channels):
                            pruned_weight[i, j] = source_shortcut_weight[output_channels[i], prev_output_channels[j]]
                
                target_state_dict[shortcut_conv_name] = pruned_weight
                
                # Handle shortcut bias if present (ResNet18 uses bias=False)
                shortcut_bias_name = f'{block_prefix}.shortcut.0.bias'
                if shortcut_bias_name in source_state_dict and shortcut_bias_name in target_shapes:
                    bias = source_state_dict[shortcut_bias_name]
                    pruned_bias = torch.zeros(target_shapes[shortcut_bias_name], device=bias.device, dtype=bias.dtype)
                    out_ch_copy = min(len(output_channels), target_shapes[shortcut_bias_name][0])
                    for i in range(out_ch_copy):
                        if i < len(output_channels):
                            pruned_bias[i] = bias[output_channels[i]]
                    target_state_dict[shortcut_bias_name] = pruned_bias
                
                # BatchNorm in shortcut (shortcut.1)
                bn_shortcut_name = f'{block_prefix}.shortcut.1'
                for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    bn_shortcut_param_name = f'{bn_shortcut_name}.{param_name}'
                    if bn_shortcut_param_name in source_state_dict and bn_shortcut_param_name in target_shapes:
                        param = source_state_dict[bn_shortcut_param_name]
                        pruned_param = torch.zeros(target_shapes[bn_shortcut_param_name], device=param.device, dtype=param.dtype)
                        out_ch_copy = min(len(output_channels), pruned_param.size(0))
                        for i in range(out_ch_copy):
                            if i < len(output_channels):
                                pruned_param[i] = param[output_channels[i]]
                        target_state_dict[bn_shortcut_param_name] = pruned_param
            
            # Update prev_output_channels for the next block/layer
            prev_output_channels = output_channels
    
    # Handle final linear layer
    linear_name = 'linear.weight'
    source_linear_weight = source_state_dict[linear_name]
    target_linear_shape = target_shapes[linear_name]
    
    # Input features depend on the output channels of the last block (prev_output_channels)
    # And the expansion factor of the block (which is 1 for BasicBlock)
    # The target shape already reflects the correct number of input features for the target model
    # Example: for target_in_planes=4, last layer has 32 output channels. 32 * 1 = 32 features.
    # Let's double-check the linear layer input size calculation in ResNet definition
    # ResNet18 linear layer input: self.in_planes (after last layer) * block.expansion
    # After layer4, self.in_planes = target_in_planes * 8
    expected_linear_input_features = (target_in_planes * 8) * 1 # Expansion is 1
    if target_linear_shape[1] != expected_linear_input_features:
         print(f"Warning: Mismatch in expected linear input features. Target shape: {target_linear_shape[1]}, Expected: {expected_linear_input_features}")
         # This might indicate an issue in the ResNet definition or the pruning logic

    pruned_linear_weight = torch.zeros(target_linear_shape, device=source_linear_weight.device, dtype=source_linear_weight.dtype)
    
    out_features = target_linear_shape[0]
    # Need to know the actual number of *kept* channels from the last conv layer
    last_block_channels = prev_output_channels # Output channels from the last block
    in_features_copy = min(len(last_block_channels), target_linear_shape[1])

    # Copy weights for the kept input features
    for i in range(out_features):
        for j in range(in_features_copy):
            if j < len(last_block_channels):
                 # Map the kept channel index from the source layer to the target linear layer
                 # This assumes the feature dimension corresponds directly to the channel index
                 source_feature_index = last_block_channels[j]
                 # Check if source_feature_index is within bounds of source linear weight
                 if source_feature_index < source_linear_weight.shape[1]:
                      pruned_linear_weight[i, j] = source_linear_weight[i, source_feature_index]
                 else:
                      # This case should ideally not happen if indices are tracked correctly
                      print(f"Warning: Source feature index {source_feature_index} out of bounds for linear layer.")

                
    target_state_dict[linear_name] = pruned_linear_weight
    
    # Handle linear bias
    linear_bias_name = 'linear.bias'
    target_state_dict[linear_bias_name] = source_state_dict[linear_bias_name]
    
    # Log what we've done
    print("\nStructured Pruning Summary:")
    print(f"Source model initial in_planes={source_in_planes}")
    print(f"Pruned model target in_planes={target_in_planes}")
    # for layer_name, indices in kept_indices.items():
    #     print(f"Layer {layer_name}: Kept {len(indices)} channels") # Optional: Detailed logging
    
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
        print("  Performing clean evaluation...")
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx % 50 == 0:
                    print(f"    Clean Eval Batch {batch_idx}/{len(data_loader)}")
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / total if total > 0 else 0
        test_acc = 100. * correct / total if total > 0 else 0
        print("  Clean evaluation completed.")
    else:
        # Adversarial evaluation
        print(f"  Performing adversarial evaluation (PGD-{steps}, eps={epsilon*255:.1f}/255)...")
        test_loss = 0
        correct = 0
        total = 0
        step_size = epsilon / 4  # Smaller step size for evaluation
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx % 10 == 0: # Print more frequently for adversarial eval
                print(f"    Adv Eval Batch {batch_idx}/{len(data_loader)}")
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize with random perturbation
            x_adv = inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # PGD attack
            for step in range(steps):
                # print(f"      PGD Step {step+1}/{steps}") # Uncomment for very detailed debugging
                x_adv.requires_grad_(True)
                outputs = model(x_adv)
                loss = F.cross_entropy(outputs, targets)
                
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                
                x_adv = x_adv.detach() + step_size * grad.sign()
                
                # Project back into epsilon ball and valid image space
                delta = torch.clamp(x_adv - inputs, -epsilon, epsilon)
                x_adv = torch.clamp(inputs + delta, 0, 1)
            
            # Final evaluation for this batch
            with torch.no_grad():
                outputs = model(x_adv)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / total if total > 0 else 0
        test_acc = 100. * correct / total if total > 0 else 0
        print(f"  Adversarial evaluation completed (PGD-{steps}, eps={epsilon*255:.1f}/255).")
    
    return test_loss, test_acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up logging
    log_path, log_data = setup_logging(args)
    
    # Get data loaders
    train_loader, test_loader, train_size, test_size = get_dataset(args)
    print(f"Train set: {train_size} samples, Test set: {test_size} samples")
    
    # Load the pretrained model (should be ResNet18 with in_planes=64)
    model = load_model(args, device)
    source_width = 64
    
    # Define the pruning stages and their corresponding arguments
    stages = ['32', '16', '4']
    stage_names = ['32', '16', '4']
    stage_iterations_args = [args.stage1_iterations, args.stage2_iterations, args.stage3_iterations]
    stage_finetune_epochs_args = [args.stage1_finetune_epochs, args.stage2_finetune_epochs, args.stage3_finetune_epochs]
    stage_clean_epochs_args = [args.stage1_clean_epochs, args.stage2_clean_epochs, args.stage3_clean_epochs]

    
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
        'source_width': source_width
    }
    
    # Initialize pruning
    current_model = model
    current_width = source_width
    
    # Process each stage sequentially
    for stage_idx, (target_width_str, stage_name) in enumerate(zip(stages, stage_names)):
        target_width = int(target_width_str)
        print(f"\n\n=== Starting STAGE {stage_idx+1}/{len(stages)}: Pruning {current_width} -> {target_width} (target in_planes={target_width}) ===\n")
        
        # Assign iterations and epochs for this stage from args
        stage_iterations = stage_iterations_args[stage_idx]
        stage_finetune_epochs = stage_finetune_epochs_args[stage_idx]
        stage_clean_epochs = stage_clean_epochs_args[stage_idx]
        
        if stage_iterations == 0: 
            print("Skipping stage - 0 iterations specified.")
            continue # Skip if no iterations assigned
        
        print(f"Running {stage_iterations} pruning iterations for this stage.")
        print(f"  Adversarial finetune epochs per iteration: {stage_finetune_epochs}")
        print(f"  Clean finetune epochs per iteration: {stage_clean_epochs}")
        
        # Calculate pruning ratio for this specific stage
        stage_prune_ratio = 1.0 - (float(target_width) / current_width)
        print(f"Stage pruning ratio: {stage_prune_ratio:.2f} ({current_width} -> {target_width})")
        
        # Iterate through pruning steps for this stage
        for iteration in range(1, stage_iterations + 1):
            print(f"\n=== Stage {stage_idx+1} Pruning Iteration {iteration}/{stage_iterations} ===")
            
            # Calculate the effective prune rate for *this* iteration
            # Aim for equal reduction steps towards the stage target
            current_prune_fraction = (stage_prune_ratio / stage_iterations)
            iteration_prune_rate = current_prune_fraction / (1 - (iteration - 1) * current_prune_fraction)
            iteration_prune_rate = max(0, min(iteration_prune_rate, 1.0)) # Clamp between 0 and 1
            
            print(f"Iteration prune rate: {iteration_prune_rate:.3f}")

            # Analyze importance for current model
            importances = analyze_layer_importance(current_model, train_loader, device)
            
            # Select channels to prune for this iteration
            prune_indices = select_channels_to_prune(importances, iteration_prune_rate)
            
            # Create a new pruned model with the target width for this stage
            pruned_model = create_pruned_model(
                current_model, 
                prune_indices, 
                device, 
                target_width=target_width_str # Pass the target width string
            )
            
            # Evaluate the pruned model before fine-tuning
            print("\n=== Evaluating pruned model (before fine-tuning) ===")
            pruned_nat_loss, pruned_nat_acc = evaluate(pruned_model, test_loader, device)
            print(f"Pruned model - Natural Accuracy: {pruned_nat_acc:.2f}% (before fine-tuning)")
            
            # Fine-tune the pruned model
            print(f"\n=== Adversarial Fine-tuning for {stage_finetune_epochs} epochs ===")
            
            optimizer = optim.SGD(
                pruned_model.parameters(),
                lr=args.lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            # Use stage-specific epochs for scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_finetune_epochs)
            
            best_adv_acc_4 = 0  # Track best adversarial accuracy at 4/255
            best_model_state = None
            
            for epoch in range(1, stage_finetune_epochs + 1):
                print(f"\nAdv Epoch {epoch}/{stage_finetune_epochs}")
                
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
                
                print(f"Adv Epoch {epoch} - Natural: {val_nat_acc:.2f}%, PGD-10 @ 1/255: {val_adv_acc_1:.2f}%, PGD-10 @ 4/255: {val_adv_acc_4:.2f}%")
                
                # Update learning rate
                scheduler.step()
                
                # Save best model based on 4/255 adversarial accuracy
                if val_adv_acc_4 > best_adv_acc_4:
                    best_adv_acc_4 = val_adv_acc_4
                    best_model_state = copy.deepcopy(pruned_model.state_dict())
                    print(f"New best adv model with adversarial accuracy {best_adv_acc_4:.2f}% at epsilon=4/255")
            
            # Restore best model from adversarial finetuning
            if best_model_state is not None:
                pruned_model.load_state_dict(best_model_state)
                
            # Additional clean fine-tuning to recover natural accuracy
            if stage_clean_epochs > 0:
                print(f"\n=== Clean Fine-tuning for {stage_clean_epochs} epochs ===")
                
                clean_optimizer = optim.SGD(
                    pruned_model.parameters(),
                    lr=args.clean_lr, 
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                )
                # Use stage-specific epochs for scheduler
                clean_scheduler = optim.lr_scheduler.CosineAnnealingLR(clean_optimizer, T_max=stage_clean_epochs)
                
                best_nat_acc = 0
                best_nat_model_state = None # Track best state during clean tuning
                current_best_adv_acc_4 = best_adv_acc_4 # Adv accuracy from previous phase

                for epoch in range(1, stage_clean_epochs + 1):
                    print(f"\nClean Epoch {epoch}/{stage_clean_epochs}")
                    
                    # Train for one epoch with clean examples
                    train_loss, train_acc = train_clean_epoch(pruned_model, train_loader, clean_optimizer, device)
                    
                    # Evaluate
                    val_nat_loss, val_nat_acc = evaluate(pruned_model, test_loader, device)
                    
                    # Evaluate on epsilon=1/255 (reporting purposes)
                    eval_eps_1 = eval_epsilons[0]  
                    _, val_adv_acc_1 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_1, steps=10)
                    
                    # Evaluate on epsilon=4/255 (for model selection constraint)
                    eval_eps_4 = 4.0/255
                    _, val_adv_acc_4 = evaluate(pruned_model, test_loader, device, epsilon=eval_eps_4, steps=10)
                    
                    print(f"Clean Epoch {epoch} - Natural: {val_nat_acc:.2f}%, PGD-10 @ 1/255: {val_adv_acc_1:.2f}%, PGD-10 @ 4/255: {val_adv_acc_4:.2f}%")
                    
                    # Update learning rate
                    clean_scheduler.step()
                    
                    # Save best model based on natural accuracy, with constraint on adversarial accuracy at 4/255
                    # Only save if natural accuracy improves AND adv accuracy doesn't drop too much from best adv epoch
                    if val_nat_acc > best_nat_acc and val_adv_acc_4 >= current_best_adv_acc_4 * 0.95: 
                        best_nat_acc = val_nat_acc
                        best_nat_model_state = copy.deepcopy(pruned_model.state_dict())
                        print(f"New best clean model state saved: Nat Acc {best_nat_acc:.2f}%, Adv Acc@4/255 {val_adv_acc_4:.2f}%")
                
                # Restore best clean model state if found
                if best_nat_model_state is not None:
                    pruned_model.load_state_dict(best_nat_model_state)
                    print(f"Restored best clean model state with Nat Acc {best_nat_acc:.2f}%")
            
            # If this is the last iteration of the stage, evaluate on all epsilons
            if iteration == stage_iterations:
                print(f"\n=== Evaluating stage {stage_idx+1} model (target in_planes={target_width}) ===")
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
            # Update current width for the next stage calculation if this is the last iteration
            if iteration == stage_iterations:
                 current_width = target_width
    
    # Save the final pruned model
    final_model = current_model[1] if isinstance(current_model, nn.Sequential) else current_model
    torch.save(final_model.state_dict(), os.path.join(args.output_dir, 'final_pruned.pth'))
    
    print("\n=== Structured pruning completed successfully ===")
    print(f"Results saved to {args.output_dir}")
    print(f"Log file: {log_path}")


if __name__ == '__main__':
    main() 
 