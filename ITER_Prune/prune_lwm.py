import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import tqdm # For progress bar during evaluation

# Add Adv-train to sys.path to import its models directly
script_dir = os.path.dirname(os.path.abspath(__file__))
adv_train_dir = os.path.join(script_dir, '../Adv-train')
sys.path.insert(0, adv_train_dir)

# Now import the WideResNet and Normalize from Adv-train
from models.wideresnet import WideResNet
from models.normalize import Normalize

# Remove Adv-train from path after import to avoid potential conflicts
sys.path.pop(0)

# We don't need the local models import anymore
# import models # Use the models from ITER_Prune 

# Remove unused import from utils.logging
# from utils.logging import parse_configs_file # Use utils from ITER_Prune

# --- GTSRB Test Dataset (copied from atas_gtsrb.py for evaluation) ---
IMAGE_SIZE = 32 # Match GTSRB processing
class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(self.csv_path):
             raise FileNotFoundError(f"CSV file not found at {self.csv_path}. Please check data_dir configuration.")
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Construct image path relative to root_dir
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['Path'])
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
             print(f"Error loading image: {img_path}")
             return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), 0

        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        label = self.df.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
# ----------------------------------------------------------------------

# --- Evaluation Function (adapted from atas_gtsrb.py) ---
def evaluate_natural(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0
    criterion_ce = nn.CrossEntropyLoss()
    
    with torch.no_grad(): # Disable gradient calculations
        pbar = tqdm.tqdm(test_loader, desc="Natural Evaluation")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion_ce(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': test_loss/len(test_loader),
                'acc': 100.*correct/total
            })
            
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Natural Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')
    return accuracy, avg_loss
# ----------------------------------------------------------

# Function to apply LWM pruning (TARGETING ONLY WideResNet part)
def apply_lwm_pruning(wrn_model, target_sparsity):
    """Applies Lowest Weight Magnitude pruning to the WRN sub-model globally
    and returns a dictionary of masks.

    Args:
        wrn_model (nn.Module): The WideResNet model part to prune.
        target_sparsity (float): The desired fraction of weights to prune (0.0 to 1.0).

    Returns:
        dict: A dictionary where keys are parameter names and values are boolean
              masks (True for kept weights, False for pruned weights).
    """
    print(f"Applying LWM pruning to target sparsity: {target_sparsity:.2f}")
    
    total_weights = 0
    magnitudes = []
    prunable_params = {} # Store references to prunable parameters

    # Collect all weight magnitudes (Conv2d and Linear layers) from WRN part
    for name, param in wrn_model.named_parameters():
        # Check the actual layer type within the submodule
        module_path = name.rsplit('.', 1)[0]
        submodule = wrn_model.get_submodule(module_path) if module_path else wrn_model
        
        if 'weight' in name and isinstance(submodule, (nn.Conv2d, nn.Linear)):
            magnitudes.append(param.data.abs().view(-1))
            total_weights += param.numel()
            prunable_params[name] = param # Keep track of param reference
            
    if not magnitudes:
        print("No weights found to prune in the WRN model.")
        return {} # Return empty mask dict

    # Concatenate all magnitudes and sort
    all_magnitudes = torch.cat(magnitudes)
    num_to_prune = int(target_sparsity * total_weights)
    
    mask_dict = {} # Dictionary to store masks
    
    if num_to_prune == 0:
        print("Target sparsity is 0, no weights pruned.")
        # Create masks indicating all weights are kept
        with torch.no_grad():
             for name, param in prunable_params.items():
                 mask_dict[name] = torch.ones_like(param.data, dtype=torch.bool)
        return mask_dict
    elif num_to_prune >= total_weights:
        print("Target sparsity >= 1.0, pruning all weights (setting to zero).")
        threshold = float('inf') # Prune everything
    else:
        # Find the threshold magnitude
        threshold, _ = torch.kthvalue(all_magnitudes, num_to_prune)
        print(f"Pruning threshold: {threshold.item()}")

    # Apply the mask (set weights below threshold to zero) and create mask dict
    pruned_count = 0
    with torch.no_grad():
        # Iterate through only the prunable parameters we identified earlier
        for name, param in prunable_params.items():
             # Generate boolean mask (True for kept weights)
             mask = param.data.abs() > threshold
             mask_dict[name] = mask # Store the boolean mask
             param.data.mul_(mask.float()) # Apply mask inplace
             pruned_count += torch.sum(~mask).item()

    actual_sparsity = pruned_count / total_weights if total_weights > 0 else 0
    print(f"Pruned {pruned_count}/{total_weights} weights. Actual Sparsity: {actual_sparsity:.4f}")
    
    # Also add masks for non-pruned parameters (e.g., biases, batchnorm) and set them to True (keep all)
    for name, param in wrn_model.named_parameters():
        if name not in mask_dict: # If not already processed (Conv/Linear weights)
            mask_dict[name] = torch.ones_like(param.data, dtype=torch.bool)

    return mask_dict

def main():
    parser = argparse.ArgumentParser(description='LWM Pruning Script with Evaluation')
    parser.add_argument('--source-net', type=str, required=True, help='Path to the model checkpoint to prune.')
    parser.add_argument('--target-sparsity', type=float, required=True, help='Target sparsity level (0.0 to 1.0).')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the pruned model checkpoint.')
    parser.add_argument('--arch', type=str, default='WideResNet', help='Base model architecture (should match Adv-train). Use WideResNet for WRN.')
    parser.add_argument('--depth', type=int, default=28, help='Depth of WideResNet.')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor of WideResNet.')
    parser.add_argument('--num-classes', type=int, default=43, help='Number of output classes (GTSRB).')
    parser.add_argument('--data-dir', type=str, default='../Adv-train/data/GTSRB', help='Path to GTSRB dataset root.')
    parser.add_argument('--test-batch-size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloader.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use for evaluation.')
    
    args = parser.parse_args()

    if not 0.0 <= args.target_sparsity <= 1.0:
        raise ValueError("Target sparsity must be between 0.0 and 1.0")
        
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model architecture: {args.arch} from Adv-train")
    # Create the full model structure including Normalize
    # Use the same normalization constants as in Adv-train/atas_gtsrb.py
    normalize_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    wrn_model = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=args.num_classes, dropRate=0.0) # Assuming dropRate=0 for eval
    model = nn.Sequential(normalize_layer, wrn_model).to(device)
    # If the saved model was wrapped in DataParallel, we might need that too, but usually state_dict is saved from model.module
    # model = nn.DataParallel(model) # Uncomment if loading fails without it

    print(f"Loading checkpoint from: {args.source_net}")
    if not os.path.isfile(args.source_net):
        raise FileNotFoundError(f"Checkpoint file not found: {args.source_net}")
        
    checkpoint = torch.load(args.source_net, map_location=device) # Load directly to target device
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model'] 
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
         raise TypeError(f"Unrecognized checkpoint format in {args.source_net}")
         
    # Adapt keys if necessary (e.g., remove 'module.' prefix from DataParallel)
    adapted_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        adapted_state_dict[name] = v
        
    # Load the state dict into the *full* sequential model
    load_result = model.load_state_dict(adapted_state_dict, strict=False) 
    print(f"Model state_dict loaded. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
    # Use strict=True if you are sure the keys match exactly
    # model.load_state_dict(adapted_state_dict, strict=True) 
    
    # --- Setup DataLoader for Evaluation ---
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Pad(4, padding_mode='reflect'), # Match atas_gtsrb.py test transform
        transforms.ToTensor(),
        # Normalization is handled by the model's Normalize layer
    ])
    test_dataset = GTSRBTestDataset(root_dir=args.data_dir, csv_file='Test.csv', transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    # ---------------------------------------

    # Evaluate *before* pruning
    print("--- Evaluating loaded model BEFORE pruning ---")
    evaluate_natural(model, device, test_loader)

    # Apply pruning *only* to the WideResNet part (model[1])
    # Get the mask dictionary back
    mask_dict = apply_lwm_pruning(model[1], args.target_sparsity)

    # Evaluate *after* pruning
    print("--- Evaluating model AFTER pruning ---")
    evaluate_natural(model, device, test_loader)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save the pruned state_dict AND the mask
    # The mask_dict keys should already match the keys in model[1].state_dict()
    save_obj = {
        'state_dict': model[1].state_dict(),
        'mask_dict': mask_dict, # Add the mask dictionary
        'arch': args.arch, # Keep base arch name for reference
        'sparsity': args.target_sparsity
    }
    torch.save(save_obj, args.output_path)
    print(f"Pruned model state_dict and mask saved to: {args.output_path}")

if __name__ == '__main__':
    main() 
