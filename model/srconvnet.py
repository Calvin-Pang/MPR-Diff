import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
    
class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        # ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        ffted = torch.view_as_real(torch.fft.fft2(x, norm='ortho'))

        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1, ) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(ffted)

        # (batch,c, t, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        output = torch.fft.ifft2(torch.view_as_complex(ffted), s=(h, w), norm='ortho').real #torch.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True)
        return output


class FConvMod(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        # my_draw_features(x.cpu().numpy(),"{}/before_fma.png".format(savepath))
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        # show_feature_map_every_channel(a.cpu().numpy(),"{}/sfi.png".format(savepath))
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        # my_draw_features(x.cpu().numpy(),"{}/after_fma.png".format(savepath))
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut
        # my_draw_features(x.cpu().numpy(),"{}/attend.png".format(savepath))
        
        return out


class KernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels, bias=True, init_weight=True):
        super().__init__()
        self.groups = groups
        self.bias = bias
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(num_kernels, dim, dim // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_kernels, dim))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        B, C, H, W = x.shape
        x = x.contiguous().view(1, B * self.dim, H, W)

        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(B * self.dim, self.dim // self.groups,
                                                               self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        else:
            x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        x = x.contiguous().view(B, self.dim, x.shape[-2], x.shape[-1])

        return x


class KernelAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_kernels=8):
        super().__init__()
        if dim != 3:
            mid_channels = dim // reduction
        else:
            mid_channels = num_kernels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, mid_channels, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, num_kernels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.sigmoid(x)
        return x


class DynamicKernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=4):
        super().__init__()
        assert dim % groups == 0
        self.attention = KernelAttention(dim, num_kernels=num_kernels)
        self.aggregation = KernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)

    def forward(self, x):
        attention = x
        attention = self.attention(attention)
        x = self.aggregation(x, attention)
        return x


class DyConv(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels=1):
        super().__init__()
        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(dim, kernel_size=kernel_size, groups=groups,
                                                 num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, num_kernels):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        x = rearrange(x, 'b c g h w -> b (c g) h w')
        x = self.proj_out(x) 
        x = x + shortcut
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kernels):
        super().__init__()
        self.attention = FConvMod(dim, num_heads)
        self.ffn = MixFFN(dim, num_kernels)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        
        return x

class PixelShuffle1D(nn.Module):
    """Custom PixelShuffle for width dimension only (1D).
    
    Args:
        upscale_factor (int): Factor to upscale the width.
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        """Forward pass for PixelShuffle1D.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor with shape (N, C//factor, H, factor * W).
        """
        N, C, H, W = x.shape
        factor = self.upscale_factor

        # Reshape the channels into the new width dimension
        x = x.view(N, C // factor, factor, H, W)
        x = x.permute(0, 1, 3, 2, 4)  # Change to (N, C//factor, H, factor, W)
        x = x.contiguous().view(N, C // factor, H, factor * W)  # Reshape to (N, C//factor, H, factor * W)

        return x
class SRNet(nn.Module):
    def __init__(self, scale, num_heads, num_kernels, colors, dim, num_blocks, rgb_range = 1.):
        super().__init__()
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.scale = scale
        self.num_heads = num_heads
        self.num_kernels = num_kernels
        self.colors = colors
        self.dim = dim
        self.num_blocks = num_blocks

        self.coord_conv = nn.Sequential(
                    # nn.Conv2d(inner_channel * 3, inner_channel * 4, kernel_size = 1),
                    # Swish(),
                    # nn.Conv2d(inner_channel * 4, 1, kernel_size = 1),
                    nn.Conv2d(3, self.dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.dim, 1, kernel_size=3, padding=1) 
                )

        self.to_feat = nn.Conv2d(self.colors, self.dim, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            *[BasicBlock(self.dim, self.num_heads, self.num_kernels) for _ in range(self.num_blocks)]
        )

        if self.scale == 8:
            self.upsampling = nn.Sequential(
                nn.Conv2d(self.dim, self.dim * 2, 1, 1, 0),
                PixelShuffle1D(2),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim * 2, 1, 1, 0),
                PixelShuffle1D(2),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim * 2, 1, 1, 0),
                PixelShuffle1D(2),
                nn.GELU()
            )
        else:
            self.upsampling = nn.Sequential(
                nn.Conv2d(self.dim, self.dim * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale),
                nn.GELU()
            )

        self.tail = nn.Conv2d(self.dim, self.colors, 3, 1, 1)

    def forward(self, x, lr_coords):
        base = x
        
        coord_features = self.coord_conv(lr_coords)
        x = x + coord_features
        # x = torch.cat([x, coord_features], dim=1)
        
        x = self.to_feat(x)
        x_init = x
        x = self.blocks(x) + x_init
        x = self.upsampling(x)
        x = self.tail(x)
        # base = F.interpolate(base, scale_factor=(1,self.scale), mode='bilinear', align_corners=False)
        return x # + base

    def load(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('upsampling') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('upsampling') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))