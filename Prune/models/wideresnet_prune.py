import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import subnet layers relative to this file's location
from .layers import SubnetConv, SubnetLinear

# Assume nn.Conv2d and nn.Linear if not provided (for potential direct use)
DEFAULT_CONV = nn.Conv2d
DEFAULT_LINEAR = nn.Linear

class BasicBlock(nn.Module):
    # Add conv_layer argument
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv_layer=DEFAULT_CONV):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # Use conv_layer
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        # Use conv_layer
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        # Use conv_layer for shortcut
        self.convShortcut = (not self.equalInOut) and conv_layer(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        # Main path
        if not self.equalInOut:
            residual = self.relu1(self.bn1(x))
        else:
            residual = self.relu1(self.bn1(x))

        residual = self.conv1(residual)
        residual = self.relu2(self.bn2(residual))

        if self.droprate > 0:
            residual = F.dropout(residual, p=self.droprate, training=self.training)

        residual = self.conv2(residual)

        # Shortcut path
        if not self.equalInOut:
            # Apply shortcut conv to the original input x if dimensions change
            # Make sure self.convShortcut is not None before calling
            if self.convShortcut is not None:
                shortcut = self.convShortcut(x)
            else:
                # This case should logically not happen if initialized correctly
                # but added for robustness / potentially satisfy linter
                shortcut = x # Fallback, although dimensions mismatch
        else:
            # Shortcut is the identity if dimensions match
            shortcut = x

        return torch.add(shortcut, residual)


class NetworkBlock(nn.Module):
    # Add conv_layer argument
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv_layer=DEFAULT_CONV):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv_layer)

    # Pass conv_layer to _make_layer and to block constructor
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv_layer):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv_layer=conv_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    # Add conv_layer and linear_layer arguments
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0, conv_layer=DEFAULT_CONV, linear_layer=DEFAULT_LINEAR):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # Use conv_layer
        self.conv1 = conv_layer(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # Pass conv_layer to NetworkBlocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv_layer=conv_layer)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv_layer=conv_layer)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv_layer=conv_layer)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # Use linear_layer
        self.fc = linear_layer(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Keep original weight init logic
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # Check against base nn.Conv2d for init
                # Weight init might need adjustment if using SubnetConv, but let's keep original for now
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): # Check against base nn.Linear for init
                m.bias.data.zero_()
        self.output_features = [] # Removed, seems unused in original

    def forward(self, x):
        # self.output_features.clear() # Removed
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # Original pooling size might be different from wrn_cifar, check pretraining
        # Assuming avg_pool2d(8) is correct based on Adv-train/models/wideresnet.py
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out) 
