from torch.nn import functional as F
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            # can do planes 32
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        # print("residual relu:", out.shape, out[0].view(-1).shape)
        return out
    
class CResNet7(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet7, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            # Calculate the linear layer size based on the final feature map size (4x4) and channels
            # For in_planes=16 -> 64 channels in final layer -> 64*4*4 = 1024
            # For in_planes=32 -> 128 channels in final layer -> 128*4*4 = 2048
            # For in_planes=64 -> 256 channels in final layer -> 256*4*4 = 4096
            final_channels = in_planes * 2  # After layer2, we have in_planes*2 channels
            self.linear1 = nn.Linear(final_channels * 4 * 4, 100)  # 4x4 is the spatial size
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out

def resnet4b():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")

def resnet4b_wide():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=32, bn=False, last_layer="dense")

def resnet4b_ultrawide():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=64, bn=False, last_layer="dense")
