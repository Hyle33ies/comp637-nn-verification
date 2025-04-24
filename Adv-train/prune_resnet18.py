#!/usr/bin/env python3
# Model Pruning Script for ResNet18 with ATAS fine-tuning

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
from models import resnet
from ATAS import test_adversarial
import adv_attack
from models.normalize import Normalize

# CIFAR10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative Pruning with Adversarial Fine-tuning')
    
    # Basic parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--arch', default='ResNet18', type=str, help='Model architecture')
    parser.add_argument('--model-dir', default='./results/cifar_atas_resnet18', type=str, help='Directory of the trained model')
    parser.add_argument('--output-dir', default='./results/cifar_atas_resnet18_pruned', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    # Training parameters
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--test-batch-size', default=128, type=int, help='Batch size for testing')
    parser.add_argument('--finetune-epochs', default=5, type=int, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for fine-tuning')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loading workers')
    
    # Pruning parameters
    parser.add_argument('--prune-percent', default=10.0, type=float, help='Percentage to prune in each iteration')
    parser.add_argument('--prune-iterations', default=9, type=int, help='Number of pruning iterations')
    parser.add_argument('--extra-iterations', default="95.0 98.0", type=str, help='Additional sparsity levels to test')
    
    # Adversarial training parameters
    parser.add_argument('--epsilon', default=8.0/255, type=float, help='Perturbation size')
    parser.add_argument('--step-size', default=2.0/255, type=float, help='Step size for PGD')
    parser.add_argument('--pgd-steps', default=10, type=int, help='Number of PGD steps for training')
    parser.add_argument('--pgd-eval-steps', default=50, type=int, help='Number of PGD steps for final evaluation')
    
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
    
    # Create model with normalization layer to match the structure in ATAS.py
    if args.arch == 'ResNet18':
        base_model = resnet.ResNet18()
        model = nn.Sequential(normalize, base_model)
    else:
        raise ValueError(f"Architecture {args.arch} is not supported")
    
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


def count_parameters(model):
    """Count the total and non-zero parameters in the model"""
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    total_params = 0
    nonzero_params = 0
    
    for name, param in base_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only count conv and linear weights
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
    
    return total_params, nonzero_params


def print_sparsity(model, log_file=None):
    """Print sparsity statistics of each layer and the overall model"""
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    total_zeros = 0
    total_elements = 0
    
    print("\nSparsity by layer:")
    for name, param in base_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only report conv and linear weights
            zeros = torch.sum(param == 0).item()
            elements = param.numel()
            sparsity = 100.0 * zeros / elements
            
            print(f"{name}: {sparsity:.2f}% sparse ({zeros}/{elements})")
            
            total_zeros += zeros
            total_elements += elements
    
    overall_sparsity = 100.0 * total_zeros / total_elements
    print(f"\nOverall model sparsity: {overall_sparsity:.2f}% ({total_zeros}/{total_elements})")
    
    return overall_sparsity


def create_mask(param, prune_percent):
    """Create a pruning mask based on weight magnitudes"""
    # Flatten the parameter
    param_flat = param.view(-1)
    
    # Get the number of weights to prune
    n_prune = int(prune_percent / 100.0 * param_flat.numel())
    
    # Get the threshold for pruning
    if n_prune > 0:
        threshold = torch.sort(torch.abs(param_flat))[0][n_prune - 1]
        
        # Create mask
        mask = torch.ones_like(param)
        mask[torch.abs(param) <= threshold] = 0
    else:
        mask = torch.ones_like(param)
    
    return mask


def prune_model(model, prune_percent, cumulative_masks=None):
    """Prune the model by removing the smallest magnitude weights"""
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    if cumulative_masks is None:
        cumulative_masks = {}
    
    for name, param in base_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only prune conv and linear weights
            # If this layer already has a mask, use it for cumulative pruning
            if name in cumulative_masks:
                current_mask = cumulative_masks[name]
                n_unpruned = torch.sum(current_mask).item()
                
                # Calculate how many more to prune
                n_to_prune = int(prune_percent / 100.0 * param.numel())
                
                # Get the non-zero weights and their indices
                param_flat = param.view(-1)
                nonzero_idxs = torch.nonzero(current_mask.view(-1), as_tuple=True)[0]
                nonzero_weights = param_flat[nonzero_idxs]
                
                # Sort the non-zero weights by magnitude
                sorted_indices = torch.argsort(torch.abs(nonzero_weights))
                n_additional = min(n_to_prune, len(sorted_indices))
                
                # Create a new mask
                new_mask = current_mask.clone()
                if n_additional > 0:
                    to_prune_idxs = nonzero_idxs[sorted_indices[:n_additional]]
                    new_mask.view(-1)[to_prune_idxs] = 0
                
                cumulative_masks[name] = new_mask
            else:
                # Create a new mask
                new_mask = create_mask(param, prune_percent)
                cumulative_masks[name] = new_mask
            
            # Apply the mask
            param.data *= cumulative_masks[name]
    
    return cumulative_masks


def apply_masks(model, masks):
    """Apply pruning masks to the model parameters"""
    # Get the base model (index 1 in Sequential)
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    
    for name, param in base_model.named_parameters():
        if name in masks:
            param.data *= masks[name]


def train_epoch(model, train_loader, optimizer, device, args, masks=None):
    """Train for one epoch with adversarial examples and maintain pruning masks"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples using adv_attack method
        # Initialize random perturbation
        x_adv = inputs + torch.zeros_like(inputs).uniform_(-args.epsilon, args.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # PGD attack
        for _ in range(args.pgd_steps):
            x_adv.requires_grad_(True)
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, targets)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            
            # Project back into epsilon ball and valid image space
            delta = torch.clamp(x_adv - inputs, -args.epsilon, args.epsilon)
            x_adv = torch.clamp(inputs + delta, 0, 1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with adversarial examples
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Apply masks to maintain pruning
        if masks is not None:
            apply_masks(model, masks)
        
        # Calculate metrics
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {train_loss/total:.4f}, "
                  f"Acc: {100.*correct/total:.2f}%, "
                  f"Memory: {memory_usage():.1f} MB")
    
    train_time = time.time() - start_time
    train_loss = train_loss / total
    train_acc = 100. * correct / total
    
    print(f"Train Time: {train_time:.2f}s, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    return train_loss, train_acc


def evaluate(model, data_loader, device, args=None, adv=False):
    """Evaluate the model on the test set, with or without adversarial examples"""
    model.eval()
    
    if not adv or args is None:
        # Use clean evaluation
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
        # Use adversarial evaluation from ATAS.py
        epsilon = args.epsilon if hasattr(args, 'epsilon') else 8.0/255
        step_size = args.step_size if hasattr(args, 'step_size') else 2.0/255
        pgd_steps = args.pgd_eval_steps if hasattr(args, 'pgd_eval_steps') else 50
        
        test_loss, test_acc = test_adversarial(
            model, data_loader, 
            epsilon=epsilon, 
            step_size=step_size,
            steps=pgd_steps
        )
    
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
    
    # Load the pretrained model
    model = load_model(args, device)
    
    # Evaluate the original model
    print("\n=== Evaluating original model ===")
    orig_natural_loss, orig_natural_acc = evaluate(model, test_loader, device)
    orig_adv_loss, orig_adv_acc = evaluate(model, test_loader, device, args, adv=True)
    
    print(f"Original model - Natural: Loss={orig_natural_loss:.4f}, Acc={orig_natural_acc:.2f}%")
    print(f"Original model - Adversarial: Loss={orig_adv_loss:.4f}, Acc={orig_adv_acc:.2f}%")
    
    # Count parameters in the original model
    total_params, nonzero_params = count_parameters(model)
    print(f"Original model parameters: {nonzero_params}/{total_params} ({100.0*nonzero_params/total_params:.2f}% dense)")
    
    # Save original model metrics
    orig_model_info = {
        'natural_acc': orig_natural_acc,
        'adv_acc': orig_adv_acc,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': 0.0,
    }
    log_data['pruning_history'].append({'iteration': 0, 'model_info': orig_model_info})
    
    # Save the original model
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    torch.save({
        'state_dict': base_model.state_dict(),
        'natural_acc': orig_natural_acc,
        'adv_acc': orig_adv_acc,
        'sparsity': 0.0,
    }, os.path.join(args.output_dir, 'original_model.pth'))
    
    # Begin pruning iterations
    print("\n=== Beginning iterative pruning ===")
    
    # Initialize variables to track cumulative masks
    cumulative_masks = {}
    current_sparsity = 0.0
    
    # Define extra sparsity levels to test
    extra_levels = [float(x) for x in args.extra_iterations.split()]
    
    # Pruning loop
    for iteration in range(1, args.prune_iterations + 1):
        print(f"\n--- Pruning Iteration {iteration}/{args.prune_iterations} ---")
        
        # Calculate current target sparsity
        target_sparsity = iteration * args.prune_percent
        print(f"Target sparsity: {target_sparsity:.1f}%")
        
        # Prune the model
        cumulative_masks = prune_model(model, args.prune_percent, cumulative_masks)
        
        # Calculate actual sparsity
        current_sparsity = print_sparsity(model)
        
        # Evaluate after pruning but before fine-tuning
        print("\nEvaluating after pruning (before fine-tuning):")
        pruned_natural_loss, pruned_natural_acc = evaluate(model, test_loader, device)
        pruned_adv_loss, pruned_adv_acc = evaluate(model, test_loader, device, args, adv=True)
        
        print(f"Pruned model - Natural: Loss={pruned_natural_loss:.4f}, Acc={pruned_natural_acc:.2f}%")
        print(f"Pruned model - Adversarial: Loss={pruned_adv_loss:.4f}, Acc={pruned_adv_acc:.2f}%")
        
        # Fine-tune the pruned model
        print(f"\nFine-tuning for {args.finetune_epochs} epochs with learning rate {args.lr}:")
        
        # Set up optimizer for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
        
        best_adv_acc = pruned_adv_acc
        best_model_state = copy.deepcopy(model.state_dict())
        
        for epoch in range(1, args.finetune_epochs + 1):
            print(f"\nEpoch {epoch}/{args.finetune_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args, cumulative_masks)
            
            # Evaluate
            val_nat_loss, val_nat_acc = evaluate(model, test_loader, device)
            val_adv_loss, val_adv_acc = evaluate(model, test_loader, device, args, adv=True)
            
            print(f"Epoch {epoch} - Natural: Loss={val_nat_loss:.4f}, Acc={val_nat_acc:.2f}%")
            print(f"Epoch {epoch} - Adversarial: Loss={val_adv_loss:.4f}, Acc={val_adv_acc:.2f}%")
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if val_adv_acc > best_adv_acc:
                best_adv_acc = val_adv_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model with adversarial accuracy {best_adv_acc:.2f}%")
        
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Evaluate best model
        print("\nEvaluating best fine-tuned model:")
        final_nat_loss, final_nat_acc = evaluate(model, test_loader, device)
        final_adv_loss, final_adv_acc = evaluate(model, test_loader, device, args, adv=True)
        
        print(f"Fine-tuned model - Natural: Loss={final_nat_loss:.4f}, Acc={final_nat_acc:.2f}%")
        print(f"Fine-tuned model - Adversarial: Loss={final_adv_loss:.4f}, Acc={final_adv_acc:.2f}%")
        
        # Count parameters after pruning
        total_params, nonzero_params = count_parameters(model)
        sparsity = 100.0 * (1 - nonzero_params/total_params)
        
        # Log results
        iteration_info = {
            'iteration': iteration,
            'target_sparsity': target_sparsity,
            'actual_sparsity': sparsity,
            'before_finetune': {
                'natural_acc': pruned_natural_acc,
                'adv_acc': pruned_adv_acc,
            },
            'after_finetune': {
                'natural_acc': final_nat_acc,
                'adv_acc': final_adv_acc,
            },
            'total_params': total_params,
            'nonzero_params': nonzero_params,
        }
        log_data['pruning_history'].append({'iteration': iteration, 'model_info': iteration_info})
        
        # Save to log file
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Save the pruned model
        base_model = model[1] if isinstance(model, nn.Sequential) else model
        torch.save({
            'state_dict': base_model.state_dict(),
            'natural_acc': final_nat_acc,
            'adv_acc': final_adv_acc,
            'sparsity': sparsity,
            'masks': cumulative_masks,
        }, os.path.join(args.output_dir, f'pruned_iteration_{iteration}.pth'))
        
        # Also save with sparsity level in filename if we hit 90%
        if iteration == args.prune_iterations:
            torch.save({
                'state_dict': base_model.state_dict(),
                'natural_acc': final_nat_acc,
                'adv_acc': final_adv_acc,
                'sparsity': sparsity,
                'masks': cumulative_masks,
            }, os.path.join(args.output_dir, f'pruned_sparsity_90.0.pth'))
    
    # Continue pruning to additional sparsity levels if specified
    for target_sparsity in extra_levels:
        print(f"\n=== Pruning to additional sparsity level: {target_sparsity}% ===")
        
        # Calculate additional pruning needed
        additional_prune = target_sparsity - current_sparsity
        if additional_prune <= 0:
            print(f"Current sparsity {current_sparsity:.2f}% already exceeds target {target_sparsity}%. Skipping.")
            continue
        
        # Performing single-shot pruning for additional level
        print(f"Performing additional pruning of {additional_prune:.2f}%")
        cumulative_masks = prune_model(model, additional_prune, cumulative_masks)
        
        # Calculate actual sparsity
        current_sparsity = print_sparsity(model)
        
        # Evaluate after pruning
        print("\nEvaluating after additional pruning (before fine-tuning):")
        pruned_natural_loss, pruned_natural_acc = evaluate(model, test_loader, device)
        pruned_adv_loss, pruned_adv_acc = evaluate(model, test_loader, device, args, adv=True)
        
        print(f"Pruned model - Natural: Loss={pruned_natural_loss:.4f}, Acc={pruned_natural_acc:.2f}%")
        print(f"Pruned model - Adversarial: Loss={pruned_adv_loss:.4f}, Acc={pruned_adv_acc:.2f}%")
        
        # Fine-tune with extended epochs for extreme sparsity
        extra_epochs = int(args.finetune_epochs * 1.5)  # More epochs for extreme pruning
        
        print(f"\nFine-tuning for {extra_epochs} epochs with learning rate {args.lr}:")
        
        # Set up optimizer for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=extra_epochs)
        
        best_adv_acc = pruned_adv_acc
        best_model_state = copy.deepcopy(model.state_dict())
        
        for epoch in range(1, extra_epochs + 1):
            print(f"\nEpoch {epoch}/{extra_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args, cumulative_masks)
            
            # Evaluate
            val_nat_loss, val_nat_acc = evaluate(model, test_loader, device)
            val_adv_loss, val_adv_acc = evaluate(model, test_loader, device, args, adv=True)
            
            print(f"Epoch {epoch} - Natural: Loss={val_nat_loss:.4f}, Acc={val_nat_acc:.2f}%")
            print(f"Epoch {epoch} - Adversarial: Loss={val_adv_loss:.4f}, Acc={val_adv_acc:.2f}%")
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if val_adv_acc > best_adv_acc:
                best_adv_acc = val_adv_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model with adversarial accuracy {best_adv_acc:.2f}%")
        
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Evaluate best model
        print("\nEvaluating best fine-tuned model:")
        final_nat_loss, final_nat_acc = evaluate(model, test_loader, device)
        final_adv_loss, final_adv_acc = evaluate(model, test_loader, device, args, adv=True)
        
        print(f"Fine-tuned model - Natural: Loss={final_nat_loss:.4f}, Acc={final_nat_acc:.2f}%")
        print(f"Fine-tuned model - Adversarial: Loss={final_adv_loss:.4f}, Acc={final_adv_acc:.2f}%")
        
        # Count parameters after pruning
        total_params, nonzero_params = count_parameters(model)
        sparsity = 100.0 * (1 - nonzero_params/total_params)
        
        # Log results for extra pruning
        extra_info = {
            'iteration': f"extra_{target_sparsity}",
            'target_sparsity': target_sparsity,
            'actual_sparsity': sparsity,
            'before_finetune': {
                'natural_acc': pruned_natural_acc,
                'adv_acc': pruned_adv_acc,
            },
            'after_finetune': {
                'natural_acc': final_nat_acc,
                'adv_acc': final_adv_acc,
            },
            'total_params': total_params,
            'nonzero_params': nonzero_params,
        }
        log_data['pruning_history'].append({'iteration': f"extra_{target_sparsity}", 'model_info': extra_info})
        
        # Save to log file
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Save the pruned model with sparsity level in filename
        base_model = model[1] if isinstance(model, nn.Sequential) else model
        torch.save({
            'state_dict': base_model.state_dict(),
            'natural_acc': final_nat_acc,
            'adv_acc': final_adv_acc,
            'sparsity': sparsity,
            'masks': cumulative_masks,
        }, os.path.join(args.output_dir, f'pruned_sparsity_{target_sparsity}.pth'))
    
    print(f"\n=== Pruning process completed successfully ===")
    print(f"Results saved to {args.output_dir}")
    print(f"Log file: {log_path}")


if __name__ == '__main__':
    main() 
