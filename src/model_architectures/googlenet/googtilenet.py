'''Modified GoogLeNet v1 from torchvision. Taken from https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py 
Reference:
 `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes

        return x


class GoogTiLeNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 4, 
        z_dim: int = 512,
        init_weights = True,
        blocks = [BasicConv2d, Inception, InceptionAux],
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        
    ) -> None:
        super().__init__()
        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        # new
        self.inception6a = inception_block(1024, 384, 192, 384, 48, 128, 128)
        self.inception6b = inception_block(1024, 384, 192, 384, 48, 128, 128)

        # after inception4a
        self.aux1 = inception_aux_block(512, z_dim, dropout=dropout_aux)
        # after inception4d
        self.aux2 = inception_aux_block(528, z_dim, dropout=dropout_aux)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        #self.fc = nn.Linear(1024, num_classes)
        self.fc = nn.Linear(1024, z_dim)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        
        aux1 = None
        if self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        
        aux2 = None
        if self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        # new
        x = self.inception6a(x)
        # N x 1024 x 7 x 7
        x = self.inception6b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x z_dim
        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> tuple:
        if self.training:
            return x, aux2, aux1
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> tuple:
        # Order of aux1 and aux2 corrected from the order in torchvision
        x, aux2, aux1 = self._forward(x)
        return self.eager_outputs(x, aux2, aux1)

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
    
    def cosine_distance(self, z_i, z_j):
        return (z_i * z_j).sum(dim=1)/(torch.linalg.norm(z_i, axis=1)*torch.linalg.norm(z_j, axis=1))
    
    def triplet_loss_cosine_similarity(self, z_p, z_n, z_d, margin=0.1, l2_reg=0):
        l_n = self.cosine_distance(z_p, z_n).exp()
        l_d = -self.cosine_distance(z_p, z_d).exp()
        l_nd = l_n + l_d
        loss = l_n + l_d + margin
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
        
        # apoorva's mod
        l2 = 0.1
        margin=0.5
        # apoorva's mod
        
        
        output_wt = 1
        aux2_wt = 0
        aux1_wt = 0
        output_patch, a2_patch, a1_patch = self.forward(patch)
        output_neighbour, a2_neighbour, a1_neighbour = self.forward(neighbor)
        output_distant, a2_distant, a1_distant = self.forward(distant)
         # apoorva's mod
        output_loss, output_l_n, output_l_d, output_l_nd = self.triplet_loss_cosine_similarity(output_patch, output_neighbour, output_distant, margin=margin, l2_reg=l2)
        aux2_loss, aux2_l_n, aux2_l_d, aux2_l_nd = self.triplet_loss_cosine_similarity(a2_patch, a2_neighbour, a2_distant, margin=margin, l2_reg=l2)
        aux1_loss, aux1_l_n, aux1_l_d, aux1_l_nd = self.triplet_loss_cosine_similarity(a1_patch, a1_neighbour, a1_distant, margin=margin, l2_reg=l2)
         # apoorva's mod
        loss = output_wt*output_loss + aux2_wt*aux2_loss + aux1_wt*aux1_loss
        l_n = output_wt*output_l_n + aux2_wt*aux2_l_n + aux1_wt*aux1_l_n
        l_d = output_wt*output_l_d + aux2_wt*aux2_l_d + aux1_wt*aux1_l_d
        l_nd = output_wt*output_l_nd + aux2_wt*aux2_l_nd + aux1_wt*aux1_l_nd
        
        return loss, l_n, l_d, l_nd


def make_googtilenet(in_channels=4, z_dim=512):
    """
    Returns a GoogTiLeNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension using the GoogLeNet architecture.
    """
    return GoogTiLeNet(in_channels=in_channels, z_dim=z_dim)