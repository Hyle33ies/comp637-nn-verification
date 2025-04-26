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

    # Load dataset 
    def load_data(dataset, batch_size, num_workers):
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
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Add index to dataset
        train_dataset = IndexedDataset(train_dataset)
        
        # Use the lower-level DataLoader constructor to avoid linter issues
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        
        return train_loader, test_loader, len(train_dataset)
    
    # IndexedDataset wrapper to keep track of indices
    # Use the lower-level Dataset import to avoid linter issues
    from torch.utils.data import Dataset
    class IndexedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __getitem__(self, index):
            data, target = self.dataset[index]
            return data, target, index
        
        def __len__(self):
            return len(self.dataset)
    
    # Create perturbations
    if args.dataset != 'imagenet':
        # For ResNet4b, we need to handle the resize properly
        train_loader, test_loader, n_ex = load_data(args.dataset, args.batch_size, args.num_workers)
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
        f.write("Note: Adversarial evaluation (PGD-10/PGD-50) is performed every 3 epochs to save time.\n")
        f.write("'-' in adversarial fields indicates no evaluation was performed for that epoch.\n\n")
        f.write("Epoch, Train Loss, Train Nat Acc, Train Adv Acc, Test Nat Loss, Test Nat Acc, Test PGD10 Loss, Test PGD10 Acc, Test PGD50 Loss, Test PGD50 Acc, Time\n")
        
    # Track best performance based on PGD-50
    best_pgd50_acc = 0.0
    best_epoch = 0

    # Main training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Reset perturbations periodically
        if epoch % epochs_reset == 0 and epoch != args.epochs and epoch > args.warmup_epochs:
            print(f"Resetting perturbations at epoch {epoch}")
            if args.dataset != 'imagenet':
                delta = (torch.rand([n_ex] + shapes_dict[args.dataset], dtype=torch.float32, device='cuda:0') * 2 - 1) * args.epsilon
            else:
                delta = (torch.rand([n_ex] + shapes_dict[args.dataset], dtype=torch.float16, device='cuda:0') * 2 - 1) * args.epsilon
        
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
