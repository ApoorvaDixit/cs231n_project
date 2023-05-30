"""
Base ResNet architecture that can be extended to create the ResNet18 and the ResNet50 architectures. 

This file is taken from https://github.com/ermongroup/tile2vec/blob/master/src/resnet.py and has been extended with different implementation choices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class BasicBlockDefault(nn.Module):
    """
    This is taken from https://github.com/ermongroup/tile2vec/blob/master/src/resnet.py.
    Different from usual BasicBlock as it has 3 conv layers.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, no_relu=False,
        activation='relu'):
        super(BasicBlock, self).__init__()
        self.no_relu = no_relu
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        # Allow downsampling in the first 3x3 conv layer.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Keep dimensions same in second 3x3 conv layer.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # no_relu layer
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        # no_relu layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        if self.no_relu:
            out = self.bn3(self.conv3(out))
            return out
        else:
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            # out = F.relu(out)
            out = self.activation_fn(out)
            return out
        
class BasicBlockCustom(nn.Module):
    """
    A basic residual block consists of 2 3x3 filters.
    """
    def __init__(self, in_channels, out_channels, stride=1, no_relu=False,
        activation='relu'):