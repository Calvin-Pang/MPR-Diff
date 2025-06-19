# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_ as kaiming_init, constant_
from torch.nn import BatchNorm2d as _BatchNorm 
import torch.nn.functional as F

def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        module (nn.Module): Module to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_(m.weight, val=1)
            constant_(m.bias, val=0)
            
def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
          
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class PixelShufflePack1D(nn.Module):
    """Pixel Shuffle upsample layer for 1D data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        
        # Conv2d to only increase the width dimension
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor,  # Only increasing along width
            kernel_size=(1, self.upsample_kernel),  # Apply kernel only along width
            padding=(0, (self.upsample_kernel - 1) // 2)  # Padding along width only
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for PixelShufflePack1DInWidth."""
        default_init_weights(self, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for 1D SR along width.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Upsampled tensor with shape (n, out_channels, h, k * w).
        """
        # Apply 2D conv to expand channels along width
        x = self.upsample_conv(x)  # Now shape (n, out_channels * scale_factor, h, w)
        
        # Reshape to apply pixel shuffle only along the width (W dimension)
        n, c, h, w = x.shape
        x = x.view(n, self.out_channels, self.scale_factor, h, w)
        x = x.permute(0, 1, 3, 2, 4)  # Change to (n, out_channels, h, scale_factor, w)
        x = x.contiguous().view(n, self.out_channels, h, self.scale_factor * w)  # Reshape to (n, out_channels, h, k * w)
        
        return x


# class PixelShufflePack(nn.Module):
#     """Pixel Shuffle upsample layer.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         scale_factor (int): Upsample ratio.
#         upsample_kernel (int): Kernel size of Conv layer to expand channels.

#     Returns:
#         Upsampled feature map.
#     """

#     def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
#                  upsample_kernel: int):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.scale_factor = scale_factor
#         self.upsample_kernel = upsample_kernel
#         self.upsample_conv = nn.Conv2d(
#             self.in_channels,
#             self.out_channels * scale_factor * scale_factor,
#             self.upsample_kernel,
#             padding=(self.upsample_kernel - 1) // 2)
#         self.init_weights()

#     def init_weights(self) -> None:
#         """Initialize weights for PixelShufflePack."""
#         default_init_weights(self, 1)

#     def forward(self, x: Tensor) -> Tensor:
#         """Forward function for PixelShufflePack.

#         Args:
#             x (Tensor): Input tensor with shape (n, c, h, w).

#         Returns:
#             Tensor: Forward results.
#         """
#         x = self.upsample_conv(x)
#         x = F.pixel_shuffle(x, self.scale_factor)
#         return x
    
class EDSRNet(BaseModule):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        rgb_mean (list[float]): Image mean in RGB orders.
            Default: [0.4488, 0.4371, 0.4040], calculated from DIV2K dataset.
        rgb_std (list[float]): Image std in RGB orders. In EDSR, it uses
            [1.0, 1.0, 1.0]. Default: [1.0, 1.0, 1.0].
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=4,
                 res_scale=1,
                 rgb_mean=[0.4488, 0.4371, 0.4040],
                 rgb_std=[1.0, 1.0, 1.0]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        self.mean = torch.Tensor(rgb_mean).view(1, -1, 1, 1)
        self.std = torch.Tensor(rgb_std).view(1, -1, 1, 1)
        self.coord_conv = nn.Sequential(
            # nn.Conv2d(inner_channel * 3, inner_channel * 4, kernel_size = 1),
            # Swish(),
            # nn.Conv2d(inner_channel * 4, 1, kernel_size = 1),
            nn.Conv2d(3, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1) 
        )
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.upsample = UpsampleModule(upscale_factor, mid_channels)
        self.conv_last = nn.Conv2d(
            mid_channels, out_channels, 3, 1, 1, bias=True)

    def forward(self, x, lr_coords):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        self.mean = self.mean.to(x)
        self.std = self.std.to(x)

        x = (x - self.mean) / self.std
        
        # coords_resized = F.interpolate(lr_coords, scale_factor = (1, 8), mode = 'bilinear', align_corners = False)
        coord_features = self.coord_conv(lr_coords)
        x = torch.cat([x, coord_features], dim=1)
        
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x * self.std + self.mean

        return x


class UpsampleModule(nn.Sequential):
    """Upsample module used in EDSR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    PixelShufflePack1D(
                        mid_channels, mid_channels, 2, upsample_kernel=3))
        elif scale == 3:
            modules.append(
                PixelShufflePack1D(
                    mid_channels, mid_channels, scale, upsample_kernel=3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')

        super().__init__(*modules)
