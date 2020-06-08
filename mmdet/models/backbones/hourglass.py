import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import BasicBlock


class HourglassModule(nn.Module):
    """ Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channel of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-module stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HourglassModule, self).__init__()

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_channel, cur_channel, cur_block, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_channel,
            next_channel,
            cur_block,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_channel,
                next_channel,
                next_block,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_channel,
            cur_channel,
            cur_block,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class HourglassNet(nn.Module):
    """ HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    arXiv: https://arxiv.org/abs/1603.06937

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-module stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times=5,
                 num_stacks=2,
                 stage_channels=[256, 256, 384, 384, 384, 512],
                 stage_blocks=[2, 2, 2, 2, 2, 4],
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HourglassNet, self).__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 128, 256, 1, stride=2, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        inter = self.stem(x)
        outs = []

        for ind, layer in enumerate(
                zip(self.hourglass_modules, self.out_convs)):
            single_hourglass, out_conv = layer

            hourglass = single_hourglass(inter)
            conv = out_conv(hourglass)
            outs.append(conv)

            if ind < self.num_stacks - 1:
                inter = self.conv1x1s[ind](inter) + self.remap_convs[ind](conv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs
