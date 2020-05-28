import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import BACKBONES
from .resnet import BasicBlock
from ..utils import ResLayer


class HGModule(nn.Module):
    """ HourGlass Module for Hourglass backbone.
        Generate module recursively and use BasicBlock as the base unit.
    """
    def __init__(self, hg_depth, stage_channels, stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HGModule, self).__init__()

        self.hg_depth = hg_depth

        curr_block = stage_blocks[0]
        next_block = stage_blocks[1]

        curr_dim = stage_channels[0]
        next_dim = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock,
            curr_dim,
            curr_dim,
            curr_block,
            norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            curr_dim,
            next_dim,
            curr_block,
            stride=2,
            norm_cfg=norm_cfg)

        self.low2 = HGModule(
            hg_depth - 1, stage_channels[1:], stage_blocks[1:]) if (
            self.hg_depth > 1) else ResLayer(
                BasicBlock,
                next_dim,
                next_dim,
                next_block,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_dim,
            curr_dim,
            curr_block,
            norm_cfg=norm_cfg,
            reverse=True)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class Hourglass(nn.Module):
    """ Hourglass backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    arXiv: https://arxiv.org/abs/1603.06937

    Args:
        hg_depth (int): Depth (also regard as downsample times) in a HGModule.
        num_stacks (int): Number of HGModule stacked, 1 for Hourglass-52,
            2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HGModule.
        stage_blocks (list[int]): Number of sub-module stacked in a HGModule.
        feat_channel (int): Feature channel of conv after a HGModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmdet.models import Hourglass
        >>> import torch
        >>> self = Hourglass()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 hg_depth=5,
                 num_stacks=2,
                 stage_channels=[256, 256, 384, 384, 384, 512],
                 stage_blocks=[2, 2, 2, 2, 2, 4],
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Hourglass, self).__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1

        curr_dim = stage_channels[0]

        self.pre = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 128, 256, 1, stride=2, norm_cfg=norm_cfg))

        self.hg_modules = nn.ModuleList([HGModule(
            hg_depth, stage_channels, stage_blocks) for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock, curr_dim, curr_dim, num_stacks - 1, norm_cfg=norm_cfg)

        self.inters_ = nn.ModuleList([
            ConvModule(curr_dim, curr_dim, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)])

        self.cnvs = nn.ModuleList([
            ConvModule(curr_dim, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)])

        self.cnvs_ = nn.ModuleList([
            ConvModule(
                feat_channel, curr_dim, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        inter = self.pre(x)
        outs = []

        for ind, layer in enumerate(zip(self.hg_modules, self.cnvs)):
            hg_, cnv_ = layer

            hg = hg_(inter)
            cnv = cnv_(hg)
            outs.append(cnv)

            if ind < self.num_stacks - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs
