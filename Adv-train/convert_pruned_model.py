#!/usr/bin/env python3
# Convert pruned model by directly zeroing out weights according to masks 
# and save only the state_dict without masks (dense model with sparse weights)
# Usage: python convert_pruned_model.py --input ./results/cifar_atas_resnet18_pruned/pruned_sparsity_90.0.pth
# Optional: --output ./results/cifar_atas_resnet18_pruned/compact_pruned_sparsity_90.0.pth
import torch
import os
import argparse

def convert_pruned_model(input_path, output_path=None):
    """
    Convert a pruned model by directly zeroing out weights according to masks 
    and save only the state_dict without masks.
    """
    if output_path is None:
        # If no output path, create one based on input file
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        output_path = os.path.join(dirname, f"compact_{basename}")
    
    print(f"Loading pruned model from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Get state dict and masks
    state_dict = checkpoint['state_dict']
    masks = checkpoint.get('masks', {})
    
    if not masks:
        print("No masks found. This might not be a pruned model.")
        # Just save the state dict if no masks
        torch.save(state_dict, output_path)
        print(f"Saved model to: {output_path}")
        return

    # Apply masks directly to weights
    original_size = sum(param.nelement() * param.element_size() 
                        for param in state_dict.values())
    
    # For each mask, apply it to the corresponding weight
    pruned_count = 0
    total_count = 0
    
    for name, mask in masks.items():
        if name in state_dict:
            # Count elements before zeroing
            total_count += state_dict[name].numel()
            pruned_count += (mask == 0).sum().item()
            
            # Directly apply mask to the weight
            state_dict[name] = state_dict[name] * mask
    
    # Calculate sparsity
    sparsity = 100.0 * pruned_count / max(total_count, 1)
    print(f"Applied masks. Sparsity: {sparsity:.2f}%")
    
    # Save only the state dict
    torch.save(state_dict, output_path)
    
    # Get the file sizes
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"Original file size: {input_size:.2f} MB")
    print(f"Compact file size: {output_size:.2f} MB")
    print(f"Size reduction: {100 * (1 - output_size/input_size):.2f}%")
    print(f"Saved compact model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pruned model by applying masks directly")
    parser.add_argument("--input", required=True, type=str, help="Path to pruned model checkpoint")
    parser.add_argument("--output", required=False, type=str, default=None, 
                        help="Output path for compact model (default: compact_<input_filename>)")
    
    args = parser.parse_args()
    convert_pruned_model(args.input, args.output)
