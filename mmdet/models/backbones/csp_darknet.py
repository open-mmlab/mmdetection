import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import BACKBONES


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Bottleneck(BaseModule):
    """Bottleneck module used in CSP layers.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 1
        with_res_shortcut (bool): Whether to use residual shortcut in blocks.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution
            in blocks. Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 with_res_shortcut=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.with_res_shortcut = \
            with_res_shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.with_res_shortcut:
            return x + out
        else:
            return out


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13)
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP block.
        out_channels (int): The output channels of the CSP block.
        expansion (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        with_res_shortcut (bool): Whether to use residual shortcut in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 num_blocks=1,
                 with_res_shortcut=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            Bottleneck(
                mid_channels,
                mid_channels,
                1.0,
                with_res_shortcut,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.blocks(x_1)

        x_2 = self.conv2(x)

        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


@BACKBONES.register_module()
class CSPDarknet(BaseModule):
    # in_channels, mid_channels, out_channels, depth_channels, stride, use_shortcut, use_spp
    arch_settings = [[64, 128, 128, 3, 2, True, False],
                     [128, 256, 256, 9, 2, True, False],
                     [256, 512, 512, 9, 2, True, False],
                     [512, 1024, 1024, 3, 2, False, True]]

    def __init__(
            self,
            deepen_factor,
            widen_factor,
            out_indices=(3, 4, 5),
            use_depthwise=False,
            arch_override=None,
            conv_cfg=None,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish'),
            init_cfg=None):
        super().__init__(init_cfg)
        if arch_override:
            self.arch_settings = arch_override
        self.out_indices = out_indices
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # stem
        self.stem = Focus(
            3,
            int(self.arch_settings[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        for i, (in_channels, mid_channels, out_channels, num_blocks, stride,
                use_shortcut, use_spp) in enumerate(self.arch_settings):
            in_channels = int(in_channels * widen_factor)
            mid_channels = int(mid_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                mid_channels,
                3,
                stride=stride,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    mid_channels,
                    mid_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                mid_channels,
                out_channels,
                num_blocks=num_blocks,
                with_res_shortcut=use_shortcut,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'dark{i + 2}', nn.Sequential(*stage))

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i in range(len(self.arch_settings)):
            stage = getattr(self, f'dark{i + 2}')
            x = stage(x)
            if i + 2 in self.out_indices:
                outs.append(x)
        return tuple(outs)
