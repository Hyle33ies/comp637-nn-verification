#!/usr/bin/env python3
import torch
import os
import sys
import shutil
from collections import OrderedDict

# Make sure we have the directory for the verification model
target_dir = "./alpha-beta-CROWN/complete_verifier/models/cifar10_resnet"
os.makedirs(target_dir, exist_ok=True)

# Source and destination paths
source_model_path = "./Adv-train/results/cifar_atas_resnet18/best.pth"
target_model_path = os.path.join(target_dir, "best.pth")

print(f"Loading source model from {source_model_path}")
if not os.path.exists(source_model_path):
    print(f"Error: Source model {source_model_path} does not exist!")
    sys.exit(1)

# Load the source model weights
source_state_dict = torch.load(source_model_path)
print(f"Loaded source model with {len(source_state_dict)} parameters")

# Preview the keys in the source model
print("\nFirst 5 keys in source model:")
for i, key in enumerate(list(source_state_dict.keys())[:5]):
    print(f"  {key} - Shape: {source_state_dict[key].shape}")

# Create new state dict for ResNet18
print("\nAttempting to convert to verification model format...")

# Option 1: Create a modified version compatible with alpha-beta-CROWN ResNet18
new_state_dict = OrderedDict()

# Manual mapping - might need adjustment based on exact source structure
# Map from 1.block1.layer.0... format to layer1.0... format
mapping = {
    # Conv1 and BN1
    "1.conv1.weight": "conv1.weight",
    "1.bn1.weight": "bn1.weight",
    "1.bn1.bias": "bn1.bias",
    "1.bn1.running_mean": "bn1.running_mean",
    "1.bn1.running_var": "bn1.running_var",
    "1.bn1.num_batches_tracked": "bn1.num_batches_tracked",
    
    # Block mappings
    # Block 1 (layer1)
    "1.block1.layer.0.conv1.weight": "layer1.0.conv1.weight",
    "1.block1.layer.0.bn1.weight": "layer1.0.bn1.weight",
    "1.block1.layer.0.bn1.bias": "layer1.0.bn1.bias",
    "1.block1.layer.0.bn1.running_mean": "layer1.0.bn1.running_mean",
    "1.block1.layer.0.bn1.running_var": "layer1.0.bn1.running_var",
    "1.block1.layer.0.bn1.num_batches_tracked": "layer1.0.bn1.num_batches_tracked",
    "1.block1.layer.0.conv2.weight": "layer1.0.conv2.weight",
    "1.block1.layer.0.bn2.weight": "layer1.0.bn2.weight",
    "1.block1.layer.0.bn2.bias": "layer1.0.bn2.bias",
    "1.block1.layer.0.bn2.running_mean": "layer1.0.bn2.running_mean",
    "1.block1.layer.0.bn2.running_var": "layer1.0.bn2.running_var",
    "1.block1.layer.0.bn2.num_batches_tracked": "layer1.0.bn2.num_batches_tracked",
    "1.block1.layer.0.convShortcut.weight": "layer1.0.shortcut.0.weight",
    
    "1.block1.layer.1.conv1.weight": "layer1.1.conv1.weight",
    "1.block1.layer.1.bn1.weight": "layer1.1.bn1.weight",
    "1.block1.layer.1.bn1.bias": "layer1.1.bn1.bias", 
    "1.block1.layer.1.bn1.running_mean": "layer1.1.bn1.running_mean",
    "1.block1.layer.1.bn1.running_var": "layer1.1.bn1.running_var",
    "1.block1.layer.1.bn1.num_batches_tracked": "layer1.1.bn1.num_batches_tracked",
    "1.block1.layer.1.conv2.weight": "layer1.1.conv2.weight",
    "1.block1.layer.1.bn2.weight": "layer1.1.bn2.weight",
    "1.block1.layer.1.bn2.bias": "layer1.1.bn2.bias",
    "1.block1.layer.1.bn2.running_mean": "layer1.1.bn2.running_mean",
    "1.block1.layer.1.bn2.running_var": "layer1.1.bn2.running_var",
    "1.block1.layer.1.bn2.num_batches_tracked": "layer1.1.bn2.num_batches_tracked",
    
    # Block 2 (layer2)
    "1.block2.layer.0.conv1.weight": "layer2.0.conv1.weight",
    "1.block2.layer.0.bn1.weight": "layer2.0.bn1.weight",
    "1.block2.layer.0.bn1.bias": "layer2.0.bn1.bias",
    "1.block2.layer.0.bn1.running_mean": "layer2.0.bn1.running_mean",
    "1.block2.layer.0.bn1.running_var": "layer2.0.bn1.running_var",
    "1.block2.layer.0.bn1.num_batches_tracked": "layer2.0.bn1.num_batches_tracked",
    "1.block2.layer.0.conv2.weight": "layer2.0.conv2.weight",
    "1.block2.layer.0.bn2.weight": "layer2.0.bn2.weight",
    "1.block2.layer.0.bn2.bias": "layer2.0.bn2.bias",
    "1.block2.layer.0.bn2.running_mean": "layer2.0.bn2.running_mean",
    "1.block2.layer.0.bn2.running_var": "layer2.0.bn2.running_var",
    "1.block2.layer.0.bn2.num_batches_tracked": "layer2.0.bn2.num_batches_tracked",
    "1.block2.layer.0.convShortcut.weight": "layer2.0.shortcut.0.weight",
    
    "1.block2.layer.1.conv1.weight": "layer2.1.conv1.weight",
    "1.block2.layer.1.bn1.weight": "layer2.1.bn1.weight", 
    "1.block2.layer.1.bn1.bias": "layer2.1.bn1.bias",
    "1.block2.layer.1.bn1.running_mean": "layer2.1.bn1.running_mean",
    "1.block2.layer.1.bn1.running_var": "layer2.1.bn1.running_var",
    "1.block2.layer.1.bn1.num_batches_tracked": "layer2.1.bn1.num_batches_tracked",
    "1.block2.layer.1.conv2.weight": "layer2.1.conv2.weight",
    "1.block2.layer.1.bn2.weight": "layer2.1.bn2.weight",
    "1.block2.layer.1.bn2.bias": "layer2.1.bn2.bias",
    "1.block2.layer.1.bn2.running_mean": "layer2.1.bn2.running_mean",
    "1.block2.layer.1.bn2.running_var": "layer2.1.bn2.running_var",
    "1.block2.layer.1.bn2.num_batches_tracked": "layer2.1.bn2.num_batches_tracked",
    
    # Block 3 (layer3)
    "1.block3.layer.0.conv1.weight": "layer3.0.conv1.weight",
    "1.block3.layer.0.bn1.weight": "layer3.0.bn1.weight",
    "1.block3.layer.0.bn1.bias": "layer3.0.bn1.bias",
    "1.block3.layer.0.bn1.running_mean": "layer3.0.bn1.running_mean",
    "1.block3.layer.0.bn1.running_var": "layer3.0.bn1.running_var",
    "1.block3.layer.0.bn1.num_batches_tracked": "layer3.0.bn1.num_batches_tracked",
    "1.block3.layer.0.conv2.weight": "layer3.0.conv2.weight",
    "1.block3.layer.0.bn2.weight": "layer3.0.bn2.weight",
    "1.block3.layer.0.bn2.bias": "layer3.0.bn2.bias",
    "1.block3.layer.0.bn2.running_mean": "layer3.0.bn2.running_mean",
    "1.block3.layer.0.bn2.running_var": "layer3.0.bn2.running_var",
    "1.block3.layer.0.bn2.num_batches_tracked": "layer3.0.bn2.num_batches_tracked",
    "1.block3.layer.0.convShortcut.weight": "layer3.0.shortcut.0.weight",
    
    "1.block3.layer.1.conv1.weight": "layer3.1.conv1.weight",
    "1.block3.layer.1.bn1.weight": "layer3.1.bn1.weight",
    "1.block3.layer.1.bn1.bias": "layer3.1.bn1.bias",
    "1.block3.layer.1.bn1.running_mean": "layer3.1.bn1.running_mean",
    "1.block3.layer.1.bn1.running_var": "layer3.1.bn1.running_var",
    "1.block3.layer.1.bn1.num_batches_tracked": "layer3.1.bn1.num_batches_tracked",
    "1.block3.layer.1.conv2.weight": "layer3.1.conv2.weight",
    "1.block3.layer.1.bn2.weight": "layer3.1.bn2.weight",
    "1.block3.layer.1.bn2.bias": "layer3.1.bn2.bias",
    "1.block3.layer.1.bn2.running_mean": "layer3.1.bn2.running_mean",
    "1.block3.layer.1.bn2.running_var": "layer3.1.bn2.running_var",
    "1.block3.layer.1.bn2.num_batches_tracked": "layer3.1.bn2.num_batches_tracked",
    
    # Final FC layer
    "1.fc.weight": "linear.weight",
    "1.fc.bias": "linear.bias"
}

# Try to map each key
for old_key, new_key in mapping.items():
    if old_key in source_state_dict:
        new_state_dict[new_key] = source_state_dict[old_key]
    else:
        print(f"Warning: Key {old_key} not found in source model")

# Check if we have all the keys we need
print(f"\nConverted {len(new_state_dict)} parameters out of {len(mapping)} in the mapping")

# Option 2: Just copy the file for now, to be safe
print(f"\nCopying original file to {target_model_path}")
shutil.copy2(source_model_path, target_model_path)

# Save the converted model if we have enough parameters
if len(new_state_dict) > 100:
    converted_model_path = os.path.join(target_dir, "converted_best.pth")
    print(f"Saving converted model to {converted_model_path}")
    torch.save(new_state_dict, converted_model_path)
    print(f"Done! You can try using this converted model with the ResNet18 architecture.")
    print(f"If that doesn't work, you'll need to modify model_defs.py to match your actual model structure.")
else:
    print("Not enough parameters were converted. Need to analyze model structure more carefully.")
    print("Please modify the mapping in this script to match your specific model.")

print("\nImportant next steps:")
print("1. If verification still fails, you'll need to check what model was ACTUALLY trained")
print("2. The model architecture in your YAML needs to match what was trained")
print("3. You may need to add a custom model definition in model_defs.py to match your trained model") 
