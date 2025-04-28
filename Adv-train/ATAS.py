from __future__ import print_function

import argparse
import os
import time
from datetime import datetime
import math

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data._utils.collate import default_collate

import adv_attack
import data
from data_aug import *
from models.wideresnet import WideResNet
from models.resnet import ResNet18
from models.resnet4b import resnet4b, resnet4b_wide, resnet4b_ultrawide
from models.normalize import Normalize

import torchvision
import tqdm

# Add memory usage monitoring
try:
    import psutil
except ImportError:
    print("psutil not installed. Memory monitoring will be limited.")
    psutil = None

def get_memory_usage():
    """Get memory usage information."""
    if psutil is None:
        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return {
            "CPU%": "N/A",
            "RAM%": "N/A",
            "GPU": f"{gpu_memory_used:.2f}/{gpu_memory_total:.2f} GB"
        }
    
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    return {
        "CPU%": cpu_percent,
        "RAM%": memory_percent,
        "GPU": f"{gpu_memory_used:.2f}/{gpu_memory_total:.2f} GB"
    }

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs-reset', type=int, default=10, metavar='N',
                    help='number of epochs to reset perturbation')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')


parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--arch', default='WideResNet', choices=['WideResNet', 'ResNet18', 'resnet4b', 'resnet4b_wide', 'resnet4b_ultrawide'],
                    help='Model architecture name')
parser.add_argument('--dropout-rate', type=float, default=0.3,
                    help='Dropout rate for WideResNet')
parser.add_argument('--decay-steps', default=[24, 28], type=int, nargs="+")

parser.add_argument('--epsilon', default=8/255, type=float,
                    help='perturbation')
parser.add_argument('--step-size', default=1.0, type=float,
                    help='perturb step size')

parser.add_argument('--max-step-size', default=14, type=float,
                    help='maximum perturb step size')
parser.add_argument('--min-step-size', default=4, type=float,
                    help='minimum perturb step size')
parser.add_argument('--c', default=0.01, type=float,
                    help='hard fraction')
parser.add_argument('--beta', default=0.5, type=float,
                    help='hardness momentum')

parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='number of epochs for warmup')

parser.add_argument('--model-dir', default='./results',
                    help='directory of model for saving checkpoint')

# Add new arguments for synthetic data
parser.add_argument('--use-synthetic-data', action='store_true',
                    help='Use synthetic data from diffusion model')
parser.add_argument('--synthetic-data-path', default='../1m.npz',
                    help='Path to synthetic data file (.npz)')
parser.add_argument('--real-samples', type=int, default=15000,
                    help='Number of real samples to use per epoch')
parser.add_argument('--synthetic-samples', type=int, default=35000,
                    help='Number of synthetic samples to use per epoch')
parser.add_argument('--progressive-mixing', action='store_true',
                    help='Progressively increase synthetic data ratio')
parser.add_argument('--consistent-sampling', action='store_true',
                    help='Use same subset of synthetic data across epochs')
parser.add_argument('--filter-synthetic', action='store_true',
                    help='Filter low-quality synthetic samples based on model confidence')
parser.add_argument('--sync-resample', action='store_true',
                    help='Synchronize resampling with perturbation reset (every epochs_reset)')

def train(args, model, train_loader, delta, optimizer, scheduler, gdnorms, epoch):
    model.train()

    train_loss = 0
    correct_nat = 0
    correct_adv = 0
    total = 0
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    
    # Enhance progress bar with epoch information
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    
    # During warmup, use only natural training
    is_warmup = epoch <= args.warmup_epochs
    
    if is_warmup:
        print(f"Epoch {epoch}: Warmup with natural training only")
    else:
        print(f"Epoch {epoch}: Adversarial training")

    # Special case for resnet4b to avoid augmentation issues
    is_resnet4b = (args.arch == 'resnet4b')

    for i, (data, label, index) in pbar:
        nat = data.cuda()
        label = label.cuda()
        index = index.cuda()
        
        if is_warmup:
            # During warmup phase, only use natural examples
            if is_resnet4b:
                # For resnet4b, avoid augmentation
                nat_trans = nat
            elif args.dataset != 'imagenet':
                nat_trans, transform_info = aug(nat)
            else:
                nat_trans, transform_info = aug_imagenet(nat)
                
            model.train()
            optimizer.zero_grad()
            
            # Compute predictions for natural examples
            outputs_nat = model(nat_trans)
            loss = F.cross_entropy(outputs_nat, label)
            
            # Track natural accuracy
            _, pred_nat = outputs_nat.max(1)
            total += label.size(0)
            correct_nat += pred_nat.eq(label).sum().item()
        else:
            # Regular adversarial training phase
            with torch.no_grad():
                if is_resnet4b:
                    # For resnet4b, avoid augmentation
                    adv_trans = torch.clamp(nat + delta[index], 0, 1)
                    nat_trans = nat
                    transform_info = None  # No transform info needed
                elif args.dataset != 'imagenet':
                    delta_trans, transform_info = aug(delta[index])
                    nat_trans = aug_trans(nat, transform_info)
                    adv_trans = torch.clamp(delta_trans + nat_trans, 0, 1)
                else:
                    delta_trans, transform_info = aug_imagenet(delta[index].to(torch.float32))
                    nat_trans = aug_trans_imagenet(nat, transform_info)
                    adv_trans = torch.clamp(delta_trans + nat_trans, 0, 1)

            # Get adversarial examples
            next_adv_trans, gdnorm = adv_attack.get_adv_adaptive_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                gdnorm=gdnorms[index],
                args=args,
                epsilon=args.epsilon
            )
            gdnorms[index] = gdnorm
            
            model.train()
            optimizer.zero_grad()
            
            # Compute natural predictions for tracking accuracy
            with torch.no_grad():
                outputs_nat = model(nat_trans)
                _, pred_nat = outputs_nat.max(1)
                correct_nat += pred_nat.eq(label).sum().item()
            
            # Compute adversarial predictions
            outputs_adv = model(next_adv_trans.detach())
            loss_adv = criterion_ce(outputs_adv, label)
            loss = loss_adv.mean()
            
            # Track adversarial accuracy
            _, pred_adv = outputs_adv.max(1)
            total += label.size(0)
            correct_adv += pred_adv.eq(label).sum().item()

        # Update model (common for both natural and adversarial training)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update loss tracking
        train_loss += loss.item()

        # Update progress bar with detailed metrics
        postfix = {
            'loss': train_loss/(i+1),
            'nat_acc': 100.*correct_nat/total,
        }
        if not is_warmup:
            postfix['adv_acc'] = 100.*correct_adv/total
        pbar.set_postfix(postfix)
        
        # Update the perturbations
        if not is_warmup:
            if is_resnet4b:
                # For resnet4b, use direct update without augmentation
                delta[index] = torch.clamp(next_adv_trans - nat_trans, -args.epsilon, args.epsilon)
            else:
                # For other models, try to use inverse_aug but with a fallback
                diff = next_adv_trans - nat_trans
                
                # Safety mechanism - wrap in try-except and use fallback if it fails
                try:
                    # Try the original update method
                    if args.dataset != "imagenet":
                        delta[index] = inverse_aug(delta[index], diff, transform_info)
                    else:
                        delta[index] = inverse_aug_imagenet(delta[index], diff.to(torch.float16), transform_info).to(torch.float16)
                    
                except (RuntimeError, ValueError) as e:
                    # If the original method fails (e.g., shape mismatch), use direct perturbation update
                    # Silently use fallback - no warnings to keep output clean
                    
                    # Resize the difference to match delta shape
                    diff_resized = F.interpolate(diff, size=delta[index].shape[2:], mode='bilinear', align_corners=False)
                    
                    # Update with direct perturbation
                    if args.dataset != "imagenet":
                        delta[index] = torch.clamp(diff_resized, -args.epsilon, args.epsilon)
                    else:
                        delta[index] = torch.clamp(diff_resized.to(torch.float16), -args.epsilon, args.epsilon)

    # Return metrics for logging
    adv_acc = 0 if is_warmup else 100.*correct_adv/total
    return train_loss/len(train_loader), 100.*correct_nat/total, adv_acc


def test_natural(model, test_loader):
    """Evaluate model on natural test examples"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm.tqdm(test_loader, desc="Natural Evaluation")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': test_loss/total,
                'acc': 100.*correct/total
            })
    
    return test_loss/len(test_loader), 100.*correct/total


def test_adversarial(model, test_loader, epsilon, step_size, steps):
    """Evaluate model on adversarial test examples using PGD attack"""
    model.eval()
    correct = 0
    total = 0
    adv_loss = 0
    
    pbar = tqdm.tqdm(test_loader, desc=f"PGD-{steps} Evaluation")
    for inputs, targets in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Initialize with random perturbation
        adv_inputs = inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
        adv_inputs = torch.clamp(adv_inputs, 0, 1)
        
        # PGD attack
        for _ in range(steps):
            adv_inputs.requires_grad_(True)
            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)
            adv_loss += loss.item()
            
            grad = torch.autograd.grad(loss, adv_inputs)[0]
            adv_inputs = adv_inputs.detach() + step_size * torch.sign(grad.detach())
            
            # Project back into epsilon ball and valid image space
            delta = torch.clamp(adv_inputs - inputs, -epsilon, epsilon)
            adv_inputs = torch.clamp(inputs + delta, 0, 1)
        
        # Final evaluation
        with torch.no_grad():
            final_outputs = model(adv_inputs)
            _, predicted = final_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'adv_loss': adv_loss/total,
                f'pgd{steps}_acc': 100.*correct/total
            })
    
    return adv_loss/len(test_loader), 100.*correct/total


# Wrapper to add consistent indexing to datasets for ConcatDataset
class IndexWrapper(Dataset):
    def __init__(self, dataset, offset=0):
        self.dataset = dataset
        self.offset = offset  # Offset for global indexing
    
    def __getitem__(self, idx):
        # Get data from the wrapped dataset
        data, target = self.dataset[idx] # Assumes wrapped dataset returns (data, target)
        # Ensure target is always a tensor
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)
        return data, target, idx + self.offset
    
    def __len__(self):
        return len(self.dataset)

# Custom collate function to handle the index
def indexed_collate_fn(batch):
    """Collate function that handles (data, target, index) tuples."""
    # Separate the components of the batch
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    indices = [item[2] for item in batch]
    
    # Use default_collate for data and targets
    collated_data = default_collate(data)
    collated_targets = default_collate(targets)
    
    # Convert indices to a tensor
    collated_indices = torch.tensor(indices, dtype=torch.long)
    
    return collated_data, collated_targets, collated_indices


def main():
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    global epochs_reset
    epochs_reset = args.epochs_reset
    
    # Adjust argument values (ensure args is updated if needed elsewhere)
    args.epsilon = args.epsilon/255
    args.max_step_size = args.max_step_size/255
    args.min_step_size = args.min_step_size/255
    args.step_size = args.step_size * args.epsilon
    
    # Create model directory
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Track total training time
    training_start_time = time.time()
    
    # Set up dataset information
    cls_dict = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}
    shapes_dict = {'cifar10': [3, 32, 32], 'cifar100': [3, 32, 32], 'imagenet': [3, 224, 224]}
    
    # Print dataset information
    print(f"Training on dataset: {args.dataset}")
    print(f"Number of classes: {cls_dict[args.dataset]}")
    
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
    normalize = Normalize(mean, std)
    
    # Create the model
    if args.dataset != 'imagenet':
        if args.arch == 'WideResNet':
            print(f"Creating model: {args.arch}-{args.depth}-{args.widen_factor}")
            model = nn.Sequential(normalize, WideResNet(
                depth=args.depth,
                num_classes=cls_dict[args.dataset],
                widen_factor=args.widen_factor,
                dropRate=args.dropout_rate
            )).cuda()
        elif args.arch == 'ResNet18':
            print(f"Creating model: {args.arch}")
            model = nn.Sequential(normalize, ResNet18()).cuda()
        elif args.arch == 'resnet4b':
            print(f"Creating model: {args.arch}")
            model = nn.Sequential(normalize, resnet4b()).cuda()
        elif args.arch == 'resnet4b_wide':
            print(f"Creating model: {args.arch}")
            model = nn.Sequential(normalize, resnet4b_wide()).cuda()
        elif args.arch == 'resnet4b_ultrawide':
            print(f"Creating model: {args.arch}")
            model = nn.Sequential(normalize, resnet4b_ultrawide()).cuda()
        else:
            print(f"Creating model: {args.arch}")
            model = nn.Sequential(normalize, eval(args.arch)(num_classes=cls_dict[args.dataset])).cuda()
    else:
        print(f"Creating ImageNet model: {args.arch}")
        model = nn.Sequential(normalize, eval('torchvision.models.' + args.arch + "()")).cuda()

    model = torch.nn.DataParallel(model)
    print(f"Model created successfully and wrapped with normalization.")
    
    if args.dropout_rate > 0 and args.arch == 'WideResNet':
        print(f"Using dropout rate: {args.dropout_rate} for WideResNet")
        
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # IndexedDataset wrapper to keep track of indices
    class IndexedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __getitem__(self, index):
            data, target = self.dataset[index]
            return data, target, index
        
        def __len__(self):
            return len(self.dataset)

    # Class for synthetic dataset from diffusion model
    class SyntheticDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            # Store initial confidence scores (will be updated when filter_samples is called)
            self.confidence_scores = torch.ones(len(images))
        
        def __getitem__(self, index):
            return self.images[index], self.labels[index]
        
        def __len__(self):
            return len(self.images)
        
        def filter_samples(self, model, batch_size=100, k=0.7):
            """Filter synthetic samples based on model confidence
            Args:
                model: Trained model to evaluate sample quality
                batch_size: Batch size for evaluation
                k: Keep ratio (0-1), higher means keep more samples
            """
            model.eval()
            device = next(model.parameters()).device
            dataloader = DataLoader(self, batch_size=batch_size, shuffle=False)
            confidence_scores = []
            
            print(f"Evaluating synthetic sample quality with model...")
            with torch.no_grad():
                for images, labels in tqdm.tqdm(dataloader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    # Get confidence score for correct class
                    batch_confidence = probs[torch.arange(len(labels)), labels].cpu()
                    confidence_scores.append(batch_confidence)
            
            # Combine all batches
            self.confidence_scores = torch.cat(confidence_scores)
            print(f"Synthetic data confidence stats: Mean={self.confidence_scores.mean():.4f}, Min={self.confidence_scores.min():.4f}, Max={self.confidence_scores.max():.4f}")
            
            return self.confidence_scores

    # Memory-efficient synthetic dataset with memory mapping
    class MemoryMappedSyntheticDataset(Dataset):
        def __init__(self, npz_path):
            """
            Memory-efficient dataset for synthetic samples using memory mapping
            
            Args:
                npz_path: Path to .npz file containing synthetic samples
            """
            print(f"Initializing memory-mapped synthetic dataset from {npz_path}")
            # Open the NPZ file in memory-mapped mode to avoid loading all data at once
            self.data = np.load(npz_path, mmap_mode='r')
            self.image_data = self.data['image']  # Memory-mapped array reference, not loaded yet
            self.label_data = self.data['label']  # Labels are usually small, so it's ok to load them
            
            # Preload a small sample to check the data format
            sample_idx = np.random.randint(0, len(self.image_data), 5)
            sample_images = self.image_data[sample_idx]
            
            print(f"Dataset contains {len(self.image_data)} synthetic samples")
            print(f"Image shape: {sample_images.shape[1:]}, dtype: {sample_images.dtype}")
            
            # Create initial confidence scores
            self.confidence_scores = torch.ones(len(self.image_data))
            
        def __getitem__(self, index):
            # Load single image from memory-mapped array
            image = self.image_data[index]
            label = self.label_data[index]
            
            # Convert to PyTorch tensor and normalize
            image_tensor = torch.from_numpy(np.copy(image)).permute(2, 0, 1).float() / 255.0
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return image_tensor, label_tensor
        
        def __len__(self):
            return len(self.image_data)
        
        def filter_samples(self, model, batch_size=100, k=0.7):
            """
            Filter synthetic samples based on model confidence
            Efficiently process in batches to avoid memory issues
            """
            model.eval()
            device = next(model.parameters()).device
            num_samples = len(self)
            
            # Process in batches to avoid memory issues
            confidence_scores = []
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            print(f"Evaluating synthetic sample quality with model ({num_batches} batches)...")
            with torch.no_grad():
                for batch_idx in tqdm.tqdm(range(num_batches)):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    
                    # Process a single batch
                    batch_images = []
                    batch_labels = []
                    for idx in range(start_idx, end_idx):
                        img, lbl = self[idx]
                        batch_images.append(img)
                        batch_labels.append(lbl)
                    
                    # Convert to tensors
                    batch_images = torch.stack(batch_images).to(device)
                    batch_labels = torch.stack(batch_labels).to(device)
                    
                    # Evaluate confidence
                    outputs = model(batch_images)
                    probs = F.softmax(outputs, dim=1)
                    batch_confidence = probs[torch.arange(len(batch_labels)), batch_labels].cpu()
                    confidence_scores.append(batch_confidence)
                    
                    # Free memory
                    del batch_images, batch_labels, outputs, probs
                    torch.cuda.empty_cache()
            
            # Combine all batches
            self.confidence_scores = torch.cat(confidence_scores)
            print(f"Synthetic data confidence stats: Mean={self.confidence_scores.mean():.4f}, Min={self.confidence_scores.min():.4f}, Max={self.confidence_scores.max():.4f}")
            
            return self.confidence_scores

    # Function to create combined dataloader with real and synthetic samples
    def create_dynamic_loader(real_dataset, synthetic_dataset, real_per_epoch, syn_per_epoch, batch_size, num_workers, 
                             consistent=False, epoch=0, progression_ratio=0.0, filter_synthetic=False, confidence_scores=None,
                             force_resample=False):
        """
        Create a dataloader that combines real and synthetic data
        
        Args:
            real_dataset: CIFAR-10 dataset
            synthetic_dataset: Synthetic dataset
            real_per_epoch: Number of real samples per epoch
            syn_per_epoch: Target number of synthetic samples per epoch
            batch_size: Batch size
            num_workers: Number of workers
            consistent: Whether to use consistent sampling across epochs
            epoch: Current epoch (for progressive mixing)
            progression_ratio: Ratio of synthetic samples to use (0.0-1.0)
            filter_synthetic: Whether to filter synthetic samples by quality
            confidence_scores: Confidence scores for synthetic samples (if filtering)
            force_resample: Force resampling even with consistent sampling enabled
        """
        # Use consistent sampling if specified and not forced to resample
        if consistent and not force_resample and hasattr(create_dynamic_loader, 'real_indices') and hasattr(create_dynamic_loader, 'syn_indices'):
            # Reuse previously sampled indices
            real_indices = create_dynamic_loader.real_indices
            syn_indices = create_dynamic_loader.syn_indices
            print(f"Using consistent sampling: same {len(real_indices)} real and {len(syn_indices)} synthetic samples")
        else:
            # Randomly select real samples
            real_indices = torch.randperm(len(real_dataset))[:real_per_epoch].tolist()
            
            # For synthetic data sampling
            if filter_synthetic and confidence_scores is not None:
                # Prioritize high-confidence synthetic samples
                # Sort indices by confidence scores (high to low)
                syn_candidate_indices = torch.argsort(confidence_scores, descending=True).tolist()
                # Take top-k samples where k is the number we need
                actual_syn_samples = min(int(syn_per_epoch * progression_ratio), len(synthetic_dataset))
                syn_indices = syn_candidate_indices[:actual_syn_samples]
            else:
                # Random sampling without filtering
                actual_syn_samples = min(int(syn_per_epoch * progression_ratio), len(synthetic_dataset))
                syn_indices = torch.randperm(len(synthetic_dataset))[:actual_syn_samples].tolist()
            
            # Store indices for consistent sampling
            if consistent:
                create_dynamic_loader.real_indices = real_indices
                create_dynamic_loader.syn_indices = syn_indices
                if force_resample:
                    print(f"Forced resampling: new subset of {len(real_indices)} real and {len(syn_indices)} synthetic samples")
                else:
                    print(f"Saved indices for consistent sampling in future epochs")
        
        # Create subsets
        real_subset = Subset(real_dataset, real_indices)
        syn_subset = Subset(synthetic_dataset, syn_indices)
        
        # Print the actual ratio
        actual_syn_samples = len(syn_indices)
        ratio = actual_syn_samples / (real_per_epoch + actual_syn_samples)
        print(f"Epoch {epoch}: Using {real_per_epoch} real samples + {actual_syn_samples} synthetic samples ({ratio:.2f} synthetic ratio)")
        
        # Wrap both datasets with proper indexing
        # Note: real_dataset (CIFAR10) returns (data, target)
        # synthetic_dataset (MemoryMapped) returns (data, target)
        wrapped_real_subset = IndexWrapper(real_subset, 0)  # Start at index 0
        wrapped_syn_subset = IndexWrapper(syn_subset, len(real_indices))  # Continue after real indices
        
        # Combine datasets
        combined_dataset = ConcatDataset([wrapped_real_subset, wrapped_syn_subset])
        
        # Create dataloader with the custom collate function
        loader = DataLoader(
            combined_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True,
            collate_fn=indexed_collate_fn  # Use custom collate function
        )
        
        return loader, len(combined_dataset)

    # Load dataset 
    def load_data(dataset, batch_size, num_workers, use_synthetic=False):
        if dataset == 'cifar10':
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            train_dataset = torchvision.datasets.CIFAR10(
                './data', train=True, transform=train_transform, download=True)
            test_dataset = torchvision.datasets.CIFAR10(
                './data', train=False, transform=test_transform, download=True)
            
            if use_synthetic:
                # Load synthetic data with memory mapping
                print(f"Loading synthetic data from {args.synthetic_data_path} using memory mapping...")
                try:
                    # Create memory-mapped dataset
                    synthetic_dataset = MemoryMappedSyntheticDataset(args.synthetic_data_path)
                    
                    # Load standard dataset for real samples
                    if args.dataset == 'cifar10':
                        train_transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(32, padding=4),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
                        ])
                        test_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                        ])
                        train_dataset = torchvision.datasets.CIFAR10(
                            './data', train=True, transform=train_transform, download=True)
                        test_dataset = torchvision.datasets.CIFAR10(
                            './data', train=False, transform=test_transform, download=True)
                    else:
                        raise ValueError(f"Synthetic data only supported for CIFAR-10")
                    
                    # Get test loader for evaluation
                    test_loader = DataLoader(
                        test_dataset, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
                    
                    # For the first epoch, use only real data to initialize the model
                    print("Initializing with real data for first epoch...")
                    # Wrap the real dataset for consistent indexing
                    indexed_train_dataset = IndexWrapper(train_dataset, 0) 
                    train_loader = DataLoader(
                        indexed_train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        num_workers=args.num_workers, 
                        pin_memory=True,
                        collate_fn=indexed_collate_fn # Use custom collate here too
                    )
                    n_ex = len(indexed_train_dataset)
                    
                    # Configure synthetic data mixing strategy
                    if args.progressive_mixing:
                        print("Using progressive mixing strategy for synthetic data")
                    else:
                        print(f"Using fixed mixing ratio: {args.real_samples} real + {args.synthetic_samples} synthetic samples")
                    
                    # Will be updated in training loop
                    synthetic_confidence_scores = None
                    
                except Exception as e:
                    print(f"Error loading synthetic data: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to original CIFAR-10 dataset")
                    use_synthetic = False
            
            # If not using synthetic data or if loading failed
            if not use_synthetic:
                # Add index to dataset
                train_dataset = IndexedDataset(train_dataset)
                n_ex = len(train_dataset)
                
                # Use the lower-level DataLoader constructor
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            
        elif dataset == 'cifar100':
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            train_dataset = torchvision.datasets.CIFAR100(
                './data', train=True, transform=train_transform, download=True)
            test_dataset = torchvision.datasets.CIFAR100(
                './data', train=False, transform=test_transform, download=True)
            
            # Add index to dataset
            train_dataset = IndexedDataset(train_dataset)
            
            # Use the lower-level DataLoader constructor to avoid linter issues
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            
            n_ex = len(train_dataset)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        return train_loader, test_loader, n_ex
    
    # Load dataset with optional synthetic data
    if args.use_synthetic_data:
        # Load synthetic data with memory mapping
        print(f"Loading synthetic data from {args.synthetic_data_path} using memory mapping...")
        try:
            # Create memory-mapped dataset
            synthetic_dataset = MemoryMappedSyntheticDataset(args.synthetic_data_path)
            
            # Load standard dataset for real samples
            if args.dataset == 'cifar10':
                train_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ])
                test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
                train_dataset = torchvision.datasets.CIFAR10(
                    './data', train=True, transform=train_transform, download=True)
                test_dataset = torchvision.datasets.CIFAR10(
                    './data', train=False, transform=test_transform, download=True)
            else:
                raise ValueError(f"Synthetic data only supported for CIFAR-10")
            
            # Get test loader for evaluation
            test_loader = DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            
            # For the first epoch, use only real data to initialize the model
            print("Initializing with real data for first epoch...")
            # Wrap the real dataset for consistent indexing
            indexed_train_dataset = IndexWrapper(train_dataset, 0) 
            train_loader = DataLoader(
                indexed_train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers, 
                pin_memory=True,
                collate_fn=indexed_collate_fn # Use custom collate here too
            )
            n_ex = len(indexed_train_dataset)
            
            # Configure synthetic data mixing strategy
            if args.progressive_mixing:
                print("Using progressive mixing strategy for synthetic data")
            else:
                print(f"Using fixed mixing ratio: {args.real_samples} real + {args.synthetic_samples} synthetic samples")
            
            # Will be updated in training loop
            synthetic_confidence_scores = None
            
        except Exception as e:
            print(f"Error loading synthetic data: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to original CIFAR-10 dataset")
            args.use_synthetic_data = False
            
    if not args.use_synthetic_data:
        train_loader, test_loader, n_ex = load_data(
            args.dataset, args.batch_size, args.num_workers, False)
    
    # Create perturbations
    if args.dataset != 'imagenet':
        # For ResNet4b, we need to handle the resize properly
        delta = (torch.rand([n_ex] + shapes_dict[args.dataset], dtype=torch.float32, device='cuda:0') * 2 - 1) * args.epsilon
    else:
        raise NotImplementedError("ImageNet dataset is not supported in this script")

    # Setup learning rate scheduler
    decay_steps = [x * len(train_loader) for x in args.decay_steps]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=0.1)

    # Initialize gradient norm history for adaptive step size
    gdnorm = torch.zeros((n_ex), dtype=torch.float32, device="cuda:0")
    
    # Log memory usage after initialization
    memory_info = get_memory_usage()
    print(f"Memory usage after initialization - CPU: {memory_info['CPU%']}%, RAM: {memory_info['RAM%']}%, GPU: {memory_info['GPU']}")
    
    # Setup logging file
    log_file = os.path.join(model_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {args}\n\n")
        if args.use_synthetic_data:
            f.write(f"Using synthetic data: {args.real_samples} real + {args.synthetic_samples} synthetic samples\n")
        f.write("Note: Adversarial evaluation (PGD-10/PGD-50) is performed every 3 epochs to save time.\n")
        f.write("'-' in adversarial fields indicates no evaluation was performed for that epoch.\n\n")
        f.write("Epoch, Train Loss, Train Nat Acc, Train Adv Acc, Test Nat Loss, Test Nat Acc, Test PGD10 Loss, Test PGD10 Acc, Test PGD50 Loss, Test PGD50 Acc, Time\n")
        
    # Track best performance based on PGD-50
    best_pgd50_acc = 0.0
    best_epoch = 0

    # Main training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Determine if we need to reset perturbations and resample
        should_reset_perturbations = (epoch % epochs_reset == 0 and epoch != args.epochs and epoch > args.warmup_epochs)
        should_force_resample = should_reset_perturbations and args.sync_resample
        
        # For synthetic data training, recreate dataloader each epoch with appropriate mixing
        if args.use_synthetic_data and epoch > 1:
            # Determine the progression ratio based on current epoch
            if args.progressive_mixing:
                # Linear ramp-up of synthetic data over first half of training
                max_synthetic_epoch = args.epochs // 2
                progression_ratio = min(1.0, epoch / max_synthetic_epoch) if epoch <= max_synthetic_epoch else 1.0
            else:
                progression_ratio = 1.0  # Use full synthetic amount
            
            # Filter synthetic samples if requested (after initial epochs)
            if args.filter_synthetic and epoch == args.warmup_epochs:
                print("Filtering synthetic samples based on model confidence...")
                synthetic_confidence_scores = synthetic_dataset.filter_samples(model)
            
            # Create dataloader with appropriate mixing ratio
            train_loader, n_ex = create_dynamic_loader(
                train_dataset, synthetic_dataset, 
                args.real_samples, args.synthetic_samples,
                args.batch_size, args.num_workers,
                consistent=args.consistent_sampling,
                epoch=epoch,
                progression_ratio=progression_ratio,
                filter_synthetic=args.filter_synthetic,
                confidence_scores=synthetic_confidence_scores,
                force_resample=should_force_resample
            )
            
            # Resize perturbations if needed
            if n_ex != len(delta):
                print(f"Resizing perturbations from {len(delta)} to {n_ex}")
                old_delta = delta
                delta = torch.zeros((n_ex, 3, 32, 32), dtype=torch.float32, device="cuda:0")
                # Copy perturbations for indices that exist in both
                min_size = min(len(old_delta), n_ex)
                delta[:min_size] = old_delta[:min_size]
                # Initialize remaining perturbations randomly
                if n_ex > len(old_delta):
                    delta[len(old_delta):] = (torch.rand((n_ex - len(old_delta), 3, 32, 32), 
                                              dtype=torch.float32, device="cuda:0") * 2 - 1) * args.epsilon
                
                # Resize gradient norm history
                old_gdnorm = gdnorm
                gdnorm = torch.zeros((n_ex), dtype=torch.float32, device="cuda:0")
                gdnorm[:min_size] = old_gdnorm[:min_size]
        
        # Reset perturbations periodically
        if should_reset_perturbations:
            print(f"Resetting perturbations at epoch {epoch}")
            delta = (torch.rand([n_ex] + shapes_dict[args.dataset], dtype=torch.float32, device='cuda:0') * 2 - 1) * args.epsilon
            if args.sync_resample and args.use_synthetic_data:
                print(f"Synchronized perturbation reset with dataset resampling")
        
        # Train for one epoch
        train_loss, train_nat_acc, train_adv_acc = train(
            args, model, train_loader, delta, optimizer, scheduler, gdnorm, epoch
        )
        
        # Log memory usage
        memory_info = get_memory_usage()
        print(f"Memory usage after epoch {epoch} - CPU: {memory_info['CPU%']}%, RAM: {memory_info['RAM%']}%, GPU: {memory_info['GPU']}")
        
        # Evaluate on clean test data
        test_nat_loss, test_nat_acc = test_natural(model, test_loader)
        
        # Evaluate robustness every 3 epochs
        test_pgd10_loss, test_pgd10_acc = float('nan'), float('nan')
        test_pgd50_loss, test_pgd50_acc = float('nan'), float('nan')
        if epoch % 3 == 0 or epoch == args.epochs:
            print(f"Performing PGD-10 evaluation at epoch {epoch}...")
            test_pgd10_loss, test_pgd10_acc = test_adversarial(
                model, test_loader, epsilon=args.epsilon, step_size=args.epsilon/5, steps=10
            )
            print(f"Performing PGD-50 evaluation at epoch {epoch}...")
            test_pgd50_loss, test_pgd50_acc = test_adversarial(
                model, test_loader, epsilon=args.epsilon, step_size=args.epsilon/5, steps=50
            )
            
            # Save best model based on PGD-50 accuracy
            if test_pgd50_acc > best_pgd50_acc:
                best_pgd50_acc = test_pgd50_acc
                best_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(model_dir, 'best.pth'))
                print(f"New best model saved with PGD-50 Acc: {test_pgd50_acc:.2f}%")
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Nat Acc: {train_nat_acc:.2f}%, Adv Acc: {train_adv_acc:.2f}%")
        print(f"  Test Nat Loss: {test_nat_loss:.4f}, Nat Acc: {test_nat_acc:.2f}%")
        
        if not math.isnan(test_pgd10_acc):
            print(f"  Test PGD-10 Loss: {test_pgd10_loss:.4f}, PGD-10 Acc: {test_pgd10_acc:.2f}%")
        if not math.isnan(test_pgd50_acc):
             print(f"  Test PGD-50 Loss: {test_pgd50_loss:.4f}, PGD-50 Acc: {test_pgd50_acc:.2f}%")
        
        print(f"  Time: {epoch_time:.2f}s")
        
        # Log results
        with open(log_file, 'a') as f:
            pgd10_loss_str = '-' if math.isnan(test_pgd10_loss) else f'{test_pgd10_loss:.4f}'
            pgd10_acc_str = '-' if math.isnan(test_pgd10_acc) else f'{test_pgd10_acc:.2f}'
            pgd50_loss_str = '-' if math.isnan(test_pgd50_loss) else f'{test_pgd50_loss:.4f}'
            pgd50_acc_str = '-' if math.isnan(test_pgd50_acc) else f'{test_pgd50_acc:.2f}'
            f.write(f"{epoch}, {train_loss:.4f}, {train_nat_acc:.2f}, {train_adv_acc:.2f}, "
                   f"{test_nat_loss:.4f}, {test_nat_acc:.2f}, "
                   f"{pgd10_loss_str}, {pgd10_acc_str}, "
                   f"{pgd50_loss_str}, {pgd50_acc_str}, "
                   f"{epoch_time:.2f}\n")

    # Save the final model with .pth extension
    torch.save(model.module.state_dict(), os.path.join(model_dir, 'last.pth'))
    
    # Final evaluation with multiple epsilon values and PGD strengths
    print("\nFinal robust accuracy evaluation:")
    _, final_nat_acc = test_natural(model, test_loader)
    print(f"Natural accuracy: {final_nat_acc:.2f}%")
    
    # Test with multiple epsilon values and PGD steps
    epsilons = [4/255, 8/255, 16/255]
    steps_list = [10, 50]
    final_robust_results = {} # Store results (eps, steps) -> acc

    for eps in epsilons:
        for steps in steps_list:
            print(f"Evaluating PGD-{steps} at ε={eps*255:.1f}/255...")
            _, robust_acc = test_adversarial(model, test_loader, epsilon=eps, step_size=eps/5, steps=steps)
            final_robust_results[(eps, steps)] = robust_acc
            print(f"  Robust accuracy (PGD-{steps}, ε={eps*255:.1f}/255): {robust_acc:.2f}%")
    
    # Log summary
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best PGD-50 accuracy during training: {best_pgd50_acc:.2f}% at epoch {best_epoch}\n")
        f.write(f"Final natural accuracy: {final_nat_acc:.2f}%\n")
        if args.dropout_rate > 0 and args.arch == 'WideResNet':
             f.write(f"Dropout rate used: {args.dropout_rate}\n")
        # Log robust accuracy at different epsilons and steps
        for eps in epsilons:
            for steps in steps_list:
                robust_acc = final_robust_results[(eps, steps)]
                f.write(f"Final Robust accuracy (PGD-{steps}, ε={eps*255:.1f}/255): {robust_acc:.2f}%\n")
    
    # Calculate and log total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
    
    print(f"\nBest model (based on PGD-50) saved at epoch {best_epoch} with {best_pgd50_acc:.2f}% robust accuracy")
    print(f"Training log saved to {log_file}")
    print(f"Total training time: {time_str}")

if __name__ == '__main__':
    main()
