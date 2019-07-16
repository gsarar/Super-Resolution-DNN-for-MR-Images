# Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        depth=5,
        wf=6,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        
        It uses bilinear upsampling in order to use less parameters, but sacrifice from 
        fine details compared to transposed convolutions as in the original paper.

        Args:
            in_channels (int): number of input channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
        """
        super(UNet, self).__init__()

       
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels,1, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())


        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())


        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=1),
        )

        self.conv_block = UNetConvBlock(in_size, out_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out