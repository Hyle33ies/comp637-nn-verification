#!/usr/bin/env python3
# Verify the sparsity level of a pruned model
# Usage: python verify_sparsity.py --model_path /root/Comp637/Adv-train/results/cifar_atas_resnet18_pruned/pruned_sparsity_98.0.pth --expected_sparsity 98.0

import torch
import argparse
import os

def verify_sparsity(model_path, expected_sparsity=None):
    """
    Verify that a model meets the expected sparsity level.
    Sparsity is defined as the percentage of zero-valued weights in the model.
    
    Args:
        model_path: Path to the model file
        expected_sparsity: Expected sparsity percentage (0-100)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different model formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            saved_sparsity = checkpoint.get('sparsity', None)
        else:
            state_dict = checkpoint
            saved_sparsity = None
            
        # Count parameters
        total_params = 0
        zero_params = 0
        
        # Only consider weights in conv and linear layers (skip biases and batch norm)
        for name, param in state_dict.items():
            if 'weight' in name and len(param.shape) > 1:  # Conv or Linear weights
                layer_total = param.numel()
                layer_zeros = (param == 0.0).sum().item()
                layer_sparsity = 100.0 * layer_zeros / layer_total
                
                total_params += layer_total
                zero_params += layer_zeros
                
                print(f"{name}: {layer_sparsity:.2f}% sparse ({layer_zeros}/{layer_total})")
        
        # Calculate overall sparsity
        actual_sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0.0
        
        print(f"\nOverall sparsity: {actual_sparsity:.4f}% ({zero_params}/{total_params})")
        
        if saved_sparsity is not None:
            print(f"Saved sparsity in model: {saved_sparsity:.4f}%")
            
        if expected_sparsity is not None:
            if actual_sparsity >= expected_sparsity:
                print(f"✅ Model meets or exceeds expected sparsity of {expected_sparsity:.2f}%")
            else:
                print(f"❌ Model does NOT meet expected sparsity of {expected_sparsity:.2f}%")
                print(f"   Shortfall: {expected_sparsity - actual_sparsity:.4f}%")
                
    except Exception as e:
        print(f"Error loading or analyzing model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the sparsity level of a pruned model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model file")
    parser.add_argument("--expected_sparsity", type=float, default=None,
                        help="Expected sparsity percentage (0-100)")
    
    args = parser.parse_args()
    verify_sparsity(args.model_path, args.expected_sparsity)
