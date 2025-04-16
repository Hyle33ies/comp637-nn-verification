import os
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
from models.normalize import Normalize

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune Wide ResNet on GTSRB')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=64, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model-dir', type=str, default='./results/gtsrb', help='directory to save model')

args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create model directory
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

# GTSRB Dataset class
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
            
        return image, label

# Data transforms
# For GTSRB, traffic signs have various sizes and we need to normalize them
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Updated size for timm model
    transforms.RandomCrop(224, padding=28),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Updated size for timm model
    transforms.ToTensor()
])

# Load datasets
train_dataset = GTSRBDataset(
    root_dir='./data/GTSRB',
    csv_file='Train.csv',
    transform=train_transform
)

test_dataset = GTSRBDataset(
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

# Count number of classes in GTSRB
meta_df = pd.read_csv('./data/GTSRB/Meta.csv')
num_classes = meta_df['ClassId'].max() + 1
print(f"Number of classes in GTSRB: {num_classes}")

# Load the model config
with open('./WRN/config.json', 'r') as f:
    config = json.load(f)

# Normalize with ImageNet stats (from config)
mean = config['pretrained_cfg']['mean']
std = config['pretrained_cfg']['std']
normalize = Normalize(mean, std)

# Create model
# Load the pretrained Wide ResNet model directly from timm
model = timm.create_model('wide_resnet50_2', pretrained=False)

# Modify the model to adjust for the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the pretrained weights from safetensors
from safetensors import safe_open
from safetensors.torch import load_file

# Load the state_dict and update only the matching keys
state_dict = load_file("./WRN/model.safetensors")
model_dict = model.state_dict()

# Filter out final fc layer from the pretrained weights
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

# Wrap model with normalization
model = nn.Sequential(normalize, model).cuda()

# Initialize optimizer and scheduler
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': train_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return train_loss/len(train_loader), 100.*correct/total

# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test')
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': test_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
    
    acc = 100.*correct/total
    print(f'Test Accuracy: {acc:.2f}%')
    
    # Save checkpoint
    if epoch % 5 == 0 or epoch == args.epochs:
        torch.save(model.state_dict(), f'{args.model_dir}/model_epoch_{epoch}.pt')
    
    return test_loss/len(test_loader), acc

# Train and evaluate
best_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f'{args.model_dir}/best.pt')
    
    print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%')

print(f'Best Test Accuracy: {best_acc:.2f}%')
# Save final model
torch.save(model.state_dict(), f'{args.model_dir}/last.pt') 
