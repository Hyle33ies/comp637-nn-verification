# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from .vgg_cifar import (
    vgg2,
    vgg2_bn,
    vgg4,
    vgg4_bn,
    vgg6,
    vgg6_bn,
    vgg8,
    vgg8_bn,
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
)
from .resnet_cifar import resnet18, resnet34, resnet50, resnet101, resnet152
from .wrn_cifar import wrn_28_10, wrn_28_1, wrn_28_4, wrn_34_10, wrn_40_2
from .basic import (
    lin_1,
    lin_2,
    lin_3,
    lin_4,
    mnist_model,
    mnist_model_large,
    cifar_model,
    cifar_model_large,
    cifar_model_resnet,
    vgg4_without_maxpool,
)

from .resnet import ResNet18, ResNet34, ResNet50
# Import the adapted WideResNet
from .wideresnet_prune import WideResNet as WideResNetPrune

# Wrapper function for the adapted WRN-28-10
def wrn_28_10_prune(conv_layer, linear_layer, **kwargs):
    # Filter out the init_type parameter if present in kwargs
    if 'init_type' in kwargs:
        # This is used by train.py but not needed by WideResNetPrune
        kwargs.pop('init_type')  
    
    # Pass the remaining kwargs to WideResNetPrune
    return WideResNetPrune(conv_layer=conv_layer, linear_layer=linear_layer, depth=28, widen_factor=10, **kwargs)

# Wrapper function for the adapted WRN-28-4 (smaller model)
def wrn_28_4_prune(conv_layer, linear_layer, **kwargs):
    # Filter out the init_type parameter if present in kwargs
    if 'init_type' in kwargs:
        # This is used by train.py but not needed by WideResNetPrune
        kwargs.pop('init_type')  
    
    # Pass the remaining kwargs to WideResNetPrune with widen_factor=4
    return WideResNetPrune(conv_layer=conv_layer, linear_layer=linear_layer, depth=28, widen_factor=4, **kwargs)

__all__ = [
    "vgg2",
    "vgg2_bn",
    "vgg4",
    "vgg4_bn",
    "vgg6",
    "vgg6_bn",
    "vgg8",
    "vgg8_bn",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wrn_28_10",
    "wrn_28_1",
    "wrn_28_4",
    "wrn_34_10",
    "wrn_40_2",
    "lin_1",
    "lin_2",
    "lin_3",
    "lin_4",
    "mnist_model",
    "mnist_model_large",
    "cifar_model",
    "cifar_model_large",
    "cifar_model_resnet",
    "vgg4_without_maxpool",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "wrn_28_10_prune",
    "wrn_28_4_prune",
]
