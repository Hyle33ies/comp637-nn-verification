import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

IMAGE_SIZE = 32 # Match the size used in pretraining

# GTSRB Dataset class (adapted from Adv-train/atas_gtsrb.py)
class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        # Use os.path.join for robust path construction
        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            # Adjust path if data_dir points directly to GTSRB folder
            # e.g., data_dir = '../Adv-train/data/GTSRB'
            # then csv should be directly inside: '../Adv-train/data/GTSRB/Train.csv'
            # If data_dir = '../Adv-train/data/', then csv is '../Adv-train/data/GTSRB/Train.csv'
            # Let's assume data_dir points to the parent of Train.csv/Test.csv for now
             raise FileNotFoundError(f"CSV file not found at {csv_path}. Please check data_dir configuration.")
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct image path relative to root_dir
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['Path'])
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            # Try adjusting path assuming root_dir is parent of 'GTSRB'
            img_path_alt = os.path.join(os.path.dirname(self.root_dir), self.df.iloc[idx]['Path'])
            try:
                 image = Image.open(img_path_alt)
            except FileNotFoundError:
                 print(f"Error loading image: {img_path} or {img_path_alt}")
                 # Return dummy data or raise error
                 return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), 0

        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.df.iloc[idx]['ClassId']

        if self.transform:
            image = self.transform(image)

        return image, label

# Main GTSRB data handling class (similar structure to Prune/data/cifar.py)
class GTSRB:
    def __init__(self, args, normalize=False):
        self.args = args
        self.root_dir = args.data_dir # Expect data_dir to point to GTSRB data root

        # Define normalization (use same values as pretraining)
        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Define transforms (use same logic as pretraining)
        self.tr_train = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Padding might be needed depending on model input expectations
            # transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(IMAGE_SIZE, padding=4), # Add RandomCrop like CIFAR
            transforms.RandomHorizontalFlip(), # Add RandomFlip like CIFAR
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # transforms.Pad(4, padding_mode='reflect'),
            transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):

        trainset = GTSRBDataset(
            root_dir=self.root_dir,
            csv_file='Train.csv',
            transform=self.tr_train
        )

        # Optional: Use SubsetRandomSampler if data_fraction < 1.0
        if self.args.data_fraction < 1.0:
             subset_indices = np.random.permutation(np.arange(len(trainset)))[
                 : int(self.args.data_fraction * len(trainset))
             ]
             sampler = SubsetRandomSampler(subset_indices.tolist()) # Cast to list[int]
             shuffle = False # Sampler handles shuffling
        else:
             sampler = None
             shuffle = True # Define shuffle when sampler is None

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True)
        )

        testset = GTSRBDataset(
            root_dir=self.root_dir,
            csv_file='Test.csv',
            transform=self.tr_test
        )

        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True)
        )

        print(
            f"Training loader: {len(sampler) if sampler else len(train_loader.dataset)} images, " # Use len(sampler) when available # type: ignore
            f"Test loader: {len(test_loader.dataset)} images" # type: ignore
        )
        return train_loader, test_loader 
