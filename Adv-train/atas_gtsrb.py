from __future__ import print_function

import argparse
import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import adv_attack
from models.normalize import Normalize
from models.wideresnet import WideResNet
from data_aug import *
import tqdm
import math
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    print("psutil not installed. Memory monitoring will be limited.")
    psutil = None

# Storage resolution for perturbations - GTSRB images are typically small
STORAGE_RESOLUTION = 32
IMAGE_SIZE = 32  # WideResNet works better with 32x32 images

# Function to upscale perturbations when needed
def upscale_perturbation(delta, target_size):
    """Upscale perturbations from storage resolution to target resolution."""
    if delta.shape[-1] == target_size:
        return delta
    return F.interpolate(delta, size=(target_size, target_size), mode='bilinear', align_corners=False)

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

# Parse arguments
parser = argparse.ArgumentParser(description='ATAS Adversarial Training for GTSRB')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
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
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--decay-steps', default=[24, 28], type=int, nargs="+")

parser.add_argument('--epsilon', default=8/255, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=1, type=int,
                    help='perturb number of steps')
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

parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='number of epochs for warmup')

parser.add_argument('--dropout-rate', type=float, default=0.3,
                    help='dropout rate for WideResNet (default: 0.3)')

parser.add_argument('--model-dir', default='./results/gtsrb_atas',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

epochs_reset = args.epochs_reset

args.epsilon = args.epsilon/255
args.max_step_size = args.max_step_size/255
args.min_step_size = args.min_step_size/255
args.step_size = args.step_size * args.epsilon

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# GTSRB Dataset class with indices for tracking perturbations
class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        label = self.df.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, idx  # Return index for perturbation lookup

# Simple Dataset class for test set (no index needed)
class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        label = self.df.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train(args, model, train_loader, delta, optimizer, scheduler, gdnorms, epoch):
    model.train()

    train_loss = 0
    correct_nat = 0
    correct_adv = 0
    total = 0
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    
    # During warmup, use only natural training
    is_warmup = epoch <= args.warmup_epochs

    for i, (data, label, index) in pbar:
        nat = data.cuda()
        label = label.cuda()
        index = index.cuda()
        
        if is_warmup:
            # During warmup, only natural training
            # For warmup, just use data augmentation on natural images
            nat_trans, transform_info = aug(nat)
            
            model.train()
            optimizer.zero_grad()
            
            # Compute predictions for natural examples
            outputs_nat = model(nat_trans)
            
            # During warmup, use only natural loss
            loss = F.cross_entropy(outputs_nat, label)
            
            # Track natural accuracy
            _, pred_nat = outputs_nat.max(1)
            total += label.size(0)
            correct_nat += pred_nat.eq(label).sum().item()
            
            # No adversarial examples during warmup
            correct_adv = 0
            
        else:
            # After warmup, do adversarial training
            # Get the perturbations for this batch and move to GPU
            batch_delta = delta[index.cpu()].cuda()
            batch_gdnorm = gdnorms[index.cpu()].cuda()
            
            # Upscale perturbation to match image size (32x32)
            batch_delta = upscale_perturbation(batch_delta, nat.shape[-1])
            
            with torch.no_grad():
                # Apply the same augmentations to natural examples and perturbations
                delta_trans, transform_info = aug(batch_delta)
                nat_trans = aug_trans(nat, transform_info)
                adv_trans = torch.clamp(delta_trans + nat_trans, 0, 1)

            # Use adaptive step size after warmup
            next_adv_trans, gdnorm = adv_attack.get_adv_adaptive_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                gdnorm=batch_gdnorm,
                args=args,
                epsilon=args.epsilon
            )
            # Update gdnorm for this batch
            gdnorms[index.cpu()] = gdnorm.cpu()
            
            model.train()
            optimizer.zero_grad()

            # Compute predictions for natural examples
            outputs_nat = model(nat_trans)
            
            # Compute predictions for adversarial examples
            outputs_adv = model(next_adv_trans.detach())
            
            # Track natural and adversarial accuracy
            _, pred_nat = outputs_nat.max(1)
            _, pred_adv = outputs_adv.max(1)
            total += label.size(0)
            correct_nat += pred_nat.eq(label).sum().item()
            correct_adv += pred_adv.eq(label).sum().item()
            
            # Compute loss on adversarial examples
            loss_adv = criterion_ce(outputs_adv, label)
            loss = loss_adv.mean()
            
            # Update persistent perturbations by inverting the augmentation
            # Downscale the perturbation before storing to save memory
            new_delta = next_adv_trans-nat_trans
            new_batch_delta = inverse_aug(batch_delta, new_delta, transform_info)
            
            # Downscale perturbation to storage resolution before saving to CPU
            if new_batch_delta.shape[-1] != STORAGE_RESOLUTION:
                new_batch_delta = F.interpolate(new_batch_delta, size=(STORAGE_RESOLUTION, STORAGE_RESOLUTION), 
                                               mode='bilinear', align_corners=False)
            
            delta[index.cpu()] = new_batch_delta.cpu()
        
        # Update model (common for both natural and adversarial training)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update loss tracking
        train_loss += loss.item()

        # Update progress bar
        postfix = {
            'loss': train_loss/(i+1),
            'nat_acc': 100.*correct_nat/total,
        }
        if not is_warmup:
            postfix['adv_acc'] = 100.*correct_adv/total
        pbar.set_postfix(postfix)

    adv_acc = 0 if is_warmup else 100.*correct_adv/total
    return train_loss/len(train_loader), 100.*correct_nat/total, adv_acc


def test_natural(model, test_loader):
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


def test_adversarial(model, test_loader, epsilon, step_size=0.007, steps=20):
    model.eval()
    correct = 0
    total = 0
    adv_loss = 0
    
    pbar = tqdm.tqdm(test_loader, desc="Adversarial Evaluation")
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
                'adv_acc': 100.*correct/total
            })
    
    return adv_loss/len(test_loader), 100.*correct/total


def main():
    # Load the class count from Meta.csv
    meta_df = pd.read_csv('./data/GTSRB/Meta.csv')
    num_classes = meta_df['ClassId'].max() + 1
    print(f"Number of classes in GTSRB: {num_classes}")
    
    # Data transforms with smaller image size
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Pad(4, padding_mode='reflect'),  # Add padding to ensure size is at least 40x40
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Pad(4, padding_mode='reflect'),  # Add padding to ensure size is at least 40x40
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = GTSRBDataset(
        root_dir='./data/GTSRB',
        csv_file='Train.csv',
        transform=train_transform
    )

    test_dataset = GTSRBTestDataset(
        root_dir='./data/GTSRB',
        csv_file='Test.csv',
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Set up normalization using CIFAR values (similar enough for GTSRB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean, std)
    
    # Create a WideResNet directly
    print("Creating WideResNet-28-10 model from scratch")
    base_model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=args.dropout_rate)
    
    # Create the full model with normalization
    model = nn.Sequential(normalize, base_model).cuda()
    print(f"Model created with {num_classes} classes")
    
    # Parallel training if multiple GPUs are available
    model = torch.nn.DataParallel(model)
    
    # Initialize optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    n_batches = len(train_loader)
    decay_steps = [x * n_batches for x in args.decay_steps]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=0.1)
    
    # Initialize persistent perturbations for all training examples at a small resolution
    n_ex = len(train_dataset)
    input_shape = [3, STORAGE_RESOLUTION, STORAGE_RESOLUTION]
    delta = (torch.rand([n_ex] + input_shape, dtype=torch.float32) * 2 - 1) * args.epsilon  # Store on CPU
    
    # Initialize gradient norm history for adaptive step size
    gdnorm = torch.zeros((n_ex), dtype=torch.float32)  # Store on CPU
    
    # Log memory usage after initializing perturbations
    memory_info = get_memory_usage()
    print(f"Memory usage after initialization - CPU: {memory_info['CPU%']}%, RAM: {memory_info['RAM%']}%, GPU: {memory_info['GPU']}")
    
    # Set up logging file
    log_file = os.path.join(model_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {args}\n\n")
        f.write("Note: Adversarial evaluation is performed every 5 epochs to save time.\n")
        f.write("'-' in adversarial fields indicates no evaluation was performed for that epoch.\n\n")
        f.write("Epoch, Train Loss, Train Nat Acc, Train Adv Acc, Test Nat Loss, Test Nat Acc, Test Adv Loss, Test Adv Acc, Time\n")
    
    # Track best performance and corresponding epoch
    best_robust_acc = 0.0
    best_epoch = 0
    
    # Log warmup status
    if args.warmup_epochs > 0:
        print(f"Using {args.warmup_epochs} epochs of natural training for warmup before adversarial training")

    if args.dropout_rate > 0:
        print(f"Using dropout rate: {args.dropout_rate}")
     
    # Main training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Reset perturbations periodically
        if epoch % epochs_reset == 0 and epoch != args.epochs and epoch > args.warmup_epochs:
            print(f"Resetting perturbations at epoch {epoch}")
            delta = (torch.rand([n_ex] + input_shape, dtype=torch.float32) * 2 - 1) * args.epsilon  # Reset on CPU
        
        # Print training mode
        if epoch <= args.warmup_epochs:
            print(f"Epoch {epoch}: Warmup with natural training only")
        else:
            print(f"Epoch {epoch}: Adversarial training") 
         
        # Train for one epoch
        train_loss, train_nat_acc, train_adv_acc = train(
            args, model, train_loader, delta, optimizer, scheduler, gdnorm, epoch
        )
        
        # Log memory usage after training
        memory_info = get_memory_usage()
        print(f"Memory usage after epoch {epoch} - CPU: {memory_info['CPU%']}%, RAM: {memory_info['RAM%']}%, GPU: {memory_info['GPU']}")
        
        # Evaluate on clean test data
        test_nat_loss, test_nat_acc = test_natural(model, test_loader)
        
        # Evaluate adversarial robustness every 3 epochs
        if epoch % 3 == 0:
            print(f"Performing adversarial evaluation at epoch {epoch}...")
            epsilon_test = args.epsilon
            test_adv_loss, test_adv_acc = test_adversarial(
                model, test_loader, epsilon=epsilon_test, step_size=epsilon_test/5, steps=20
            )
        else:
            # Skip adversarial evaluation for other epochs
            test_adv_loss, test_adv_acc = float('nan'), float('nan')
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Nat Acc: {train_nat_acc:.2f}%, Adv Acc: {train_adv_acc:.2f}%")
        print(f"  Test Nat Loss: {test_nat_loss:.4f}, Nat Acc: {test_nat_acc:.2f}%")
        
        if not math.isnan(test_adv_acc):
            print(f"  Test Adv Loss: {test_adv_loss:.4f}, Adv Acc: {test_adv_acc:.2f}%")
        
        print(f"  Time: {epoch_time:.2f}s")
        
        # Log results
        with open(log_file, 'a') as f:
            f.write(f"{epoch}, {train_loss:.4f}, {train_nat_acc:.2f}, {train_adv_acc:.2f}, "
                   f"{test_nat_loss:.4f}, {test_nat_acc:.2f}, "
                   f"{'-' if math.isnan(test_adv_loss) else f'{test_adv_loss:.4f}'}, "
                   f"{'-' if math.isnan(test_adv_acc) else f'{test_adv_acc:.2f}'}, "
                   f"{epoch_time:.2f}\n")
        
        # Save checkpoint at regular intervals
        # if epoch % 10 == 0:
        #     torch.save(model.module.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pt'))
        
        # Keep track of best model - only update for epochs where we actually evaluated adversarial performance
        if not math.isnan(test_adv_acc) and test_adv_acc > best_robust_acc: # best robustness
            best_robust_acc = test_adv_acc
            best_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(model_dir, 'best.pt'))
            
    # Save the final model
    torch.save(model.module.state_dict(), os.path.join(model_dir, 'last.pt'))
    
    # Final evaluation with stronger attack
    print("\nFinal robust accuracy evaluation:")
    _, final_nat_acc = test_natural(model, test_loader)
    print(f"Natural accuracy: {final_nat_acc:.2f}%")
    
    # Test with multiple epsilon values
    epsilons = [4/255, 8/255, 16/255]
    for eps in epsilons:
        _, robust_acc = test_adversarial(model, test_loader, epsilon=eps, step_size=eps/5, steps=20)
        print(f"Robust accuracy (ε={eps*255:.1f}/255): {robust_acc:.2f}%")
    
    # Log summary
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best robust accuracy: {best_robust_acc:.2f}% at epoch {best_epoch}\n")
        f.write(f"Final natural accuracy: {final_nat_acc:.2f}%\n")
        f.write(f"Dropout rate used: {args.dropout_rate}\n")
        for eps in epsilons:
            _, robust_acc = test_adversarial(model, test_loader, epsilon=eps, step_size=eps/5, steps=20)
            f.write(f"Robust accuracy (ε={eps*255:.1f}/255): {robust_acc:.2f}%\n")
    
    print(f"\nBest model saved at epoch {best_epoch} with {best_robust_acc:.2f}% robust accuracy")
    print(f"Training log saved to {log_file}")

if __name__ == '__main__':
    main() 
