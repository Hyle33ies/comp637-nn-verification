import torch
import torch.nn as nn
import torchvision

import os
import math
import numpy as np

# TODO: avoid freezing bn_params
# Some utils are borrowed from https://github.com/allenai/hidden-networks
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name, freeze_bn=False):
    """
    unfreeze vars. If freeze_bn then only unfreeze batch_norm params.
    """
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)


def show_gradients(model):
    """
    Print gradients of parameters in the model. To be used for debugging.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def prepare_model(model, args):
    """
    Prepare model for pruning/finetuning. Freeze/unfreeze parameters based on exp_mode.
    """

    if args.exp_mode == "finetune":
        # In finetune mode, all scoring parameters are frozen. Weight and bias are trained.
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        # freeze_vars(model, "popup_scores") # No popup_scores in LWM

    elif args.exp_mode == "prune":
        # In prune mode, only scoring parameters are trained. Weight and bias are frozen.
        freeze_vars(model, "weight")
        freeze_vars(model, "bias", args.freeze_bn)
        # unfreeze_vars(model, "popup_scores") # No popup_scores in LWM

    elif args.exp_mode == "pretrain":
        # In pretrain mode, only weight and bias are trained.
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        # freeze_vars(model, "popup_scores") # No popup_scores in LWM


def dense_to_subnet(model, state_dict):
    """
        Load a dict with dense-layer in a model trained with subnet layers. 
    """
    model.load_state_dict(state_dict, strict=False)


def sanity_check_paramter_updates(model, last_ckpt):
    """
    Check if parameters are updated.
    Prints the norm of the change in parameters.
    """
    weights_updated = False
    scores_updated = False

    for ((n1, p1), (n2, p2)) in zip(
        model.named_parameters(), last_ckpt.items()
    ):
        if p1.requires_grad:
            change = torch.norm(p1.data - p2.data)
            if change > 0:
                if "popup_scores" in n1:
                    scores_updated = True
                else:
                    weights_updated = True
                # print(n1, change)

    return weights_updated, scores_updated
