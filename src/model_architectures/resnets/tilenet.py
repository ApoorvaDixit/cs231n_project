'''Modified ResNet-18 in PyTorch. Taken from https://github.com/ermongroup/tile2vec/blob/master/src/tilenet.py.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, no_relu=False,
        activation='relu'):
        super(BasicBlock, self).__init__()
        self.no_relu = no_relu
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # no_relu layer
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        # no_relu layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
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


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        """
        Input:
            num_blocks: num_blocks[i] represents how many Residual blocks will be concatenated in sequence within layer i.
            in_channels: number of channels in unprocessed input representation.
            z_dim: number of channels that should be in the output.
        """
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        # Stride is applied in first convolution of first block only.
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)

    def _make_layer(self, block, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride,
                no_relu=no_relu, activation=self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x):
        x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2_reg=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2_reg != 0:
            loss += l2_reg * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        patch.to(self.device)
        neighbor.to(self.device)
        distant.to(self.device)
        
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2_reg=l2)


def make_tilenet_18(in_channels=4, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim)