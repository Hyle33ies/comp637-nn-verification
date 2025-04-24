#!/usr/bin/env python3
import torch
import os
import sys
import shutil
from collections import OrderedDict

# Make sure we have the directory for the verification model
target_dir = "./alpha-beta-CROWN/complete_verifier/models/cifar10_resnet"
# target_dir = "./ITER_Prune/results/cifar10_iter_lwm_resnet18"
os.makedirs(target_dir, exist_ok=True)

# Source and destination paths
# source_model_path = "./Adv-train/results/cifar_atas_resnet18/best.pth"
# target_model_path = os.path.join(target_dir, "best.pth")
source_model_path = "./Adv-train/results/cifar_atas_resnet18_pruned/compact_pruned_sparsity_98.0.pth"
target_model_path = os.path.join(target_dir, "best.pth")

print(f"Loading source model from {source_model_path}")
if not os.path.exists(source_model_path):
    print(f"Error: Source model {source_model_path} does not exist!")
    sys.exit(1)

# Load the source model weights
source_state_dict = torch.load(source_model_path)
print(f"Loaded source model with {len(source_state_dict)} parameters")

# Preview the keys in the source model
print("\nFirst 10 keys in source model:")
for i, key in enumerate(list(source_state_dict.keys())[:10]):
    print(f"  {key} - Shape: {source_state_dict[key].shape}")

# Create new state dict for ResNet18
print("\nAttempting to convert to verification model format...")

# First, let's check if there are any layer1, layer2, etc. keys in the source model
has_layer_keys = any(key.startswith(('layer1', 'layer2', 'layer3', 'layer4')) for key in source_state_dict.keys())
if has_layer_keys:
    print("Source model already has layer-style keys. No remapping needed, just copying...")
    new_state_dict = source_state_dict
else:
    # New mapping based on the actual key structure (assuming from check_model_compatibility.py output)
    new_state_dict = OrderedDict()

    # Define mapping for simple 1.x keys to standard ResNet keys
    standard_mapping = {}
    
    # Check all keys in source_state_dict
    all_keys = list(source_state_dict.keys())
    print("\nAnalyzing all keys in source model...")
    
    # Determine mapping pattern by analyzing key prefixes
    if all_keys and all_keys[0].startswith('1.'):
        print("Detected '1.' prefix pattern in keys")
        
        # Map direct keys
        standard_mapping = {
            # Conv1 and BN1
            "1.conv1.weight": "conv1.weight",
            "1.bn1.weight": "bn1.weight",
            "1.bn1.bias": "bn1.bias",
            "1.bn1.running_mean": "bn1.running_mean",
            "1.bn1.running_var": "bn1.running_var",
            "1.bn1.num_batches_tracked": "bn1.num_batches_tracked",
        }
        
        # Map layer keys - look for patterns like 1.layer1, 1.layer2, etc.
        for i in range(1, 5):  # ResNet usually has 4 layer groups
            if any(f"1.layer{i}." in key for key in all_keys):
                print(f"Found 1.layer{i} pattern")
                for key in all_keys:
                    if key.startswith(f"1.layer{i}."):
                        # Replace '1.layerX.' with 'layerX.'
                        new_key = key.replace(f"1.layer{i}.", f"layer{i}.")
                        standard_mapping[key] = new_key
            # Also check for alternate patterns like 1.blockX
            elif any(f"1.block{i}." in key for key in all_keys):
                print(f"Found 1.block{i} pattern")
                for key in all_keys:
                    if key.startswith(f"1.block{i}."):
                        # Need to map block structure to layer structure
                        # This might need customization based on exact structure
                        new_key = key.replace(f"1.block{i}.", f"layer{i}.")
                        standard_mapping[key] = new_key
        
        # Map FC/Linear layer
        if "1.fc.weight" in all_keys:
            standard_mapping["1.fc.weight"] = "linear.weight"
            standard_mapping["1.fc.bias"] = "linear.bias"
        elif "1.linear.weight" in all_keys:
            standard_mapping["1.linear.weight"] = "linear.weight"
            standard_mapping["1.linear.bias"] = "linear.bias"
        
        # Handle any keys that haven't been mapped
        for key in all_keys:
            if key not in standard_mapping:
                print(f"Warning: No mapping defined for key: {key}")
                # Create a default mapping by removing the '1.' prefix
                if key.startswith('1.'):
                    if 'fc.' in key:
                        # Convert fc to linear for compatibility
                        new_key = key[2:].replace('fc.', 'linear.')
                    else:
                        new_key = key[2:]  # Remove '1.' prefix
                    standard_mapping[key] = new_key
                    print(f"  Auto-mapping: {key} -> {new_key}")
    
    # Apply the mapping
    print("\nApplying key mapping...")
    for old_key, new_key in standard_mapping.items():
        if old_key in source_state_dict:
            new_state_dict[new_key] = source_state_dict[old_key]
            print(f"  Mapped: {old_key} -> {new_key}")
        else:
            print(f"  Warning: Key {old_key} not found in source model")

# Check how many parameters we mapped
print(f"\nConverted {len(new_state_dict)} parameters out of {len(source_state_dict)} in the source model")

# Save both the original and converted models
print(f"\nCopying original file to {target_model_path}")
shutil.copy2(source_model_path, target_model_path)

# Save the converted model if we have enough parameters
if len(new_state_dict) > 0.9 * len(source_state_dict):  # At least 90% of params mapped
    converted_model_path = os.path.join(target_dir, "converted_best.pth")
    print(f"Saving converted model to {converted_model_path}")
    torch.save(new_state_dict, converted_model_path)
    print(f"Done! You can try using this converted model with the verification architecture.")
else:
    print("Not enough parameters were converted. Need to analyze model structure more carefully.")
    print("Key structure in source model doesn't match expected patterns.")
    
    # Save whatever we have anyway, might be useful for debugging
    if len(new_state_dict) > 0:
        partial_path = os.path.join(target_dir, "partial_converted_best.pth")
        print(f"Saving partially converted model to {partial_path} for analysis")
        torch.save(new_state_dict, partial_path)

print("\nImportant next steps:")
print("1. If verification still fails, you'll need to check what model was ACTUALLY trained")
print("2. The model architecture in your YAML needs to match what was trained")
print("3. You may need to add a custom model definition in model_defs.py to match your trained model") 
print("4. Alternatively, modify your training code to save with the expected key format") 
