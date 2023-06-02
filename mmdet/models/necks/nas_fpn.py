# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.merge_cells import GlobalPoolingCell, SumCell
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


@MODELS.register_module()
class NASFPN(BaseModule):
    """NAS-FPN.

    Implementation of `NAS-FPN: Learning Scalable Feature Pyramid Architecture
    for Object Detection <https://arxiv.org/abs/1904.07392>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        stack_times: int,
        start_level: int = 0,
        end_level: int = -1,
        norm_cfg: OptConfigType = None,
        init_cfg: MultiConfig = dict(type='Caffe2Xavier', layer='Conv2d')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.stack_times = stack_times
        self.norm_cfg = norm_cfg

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(
                out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            self.extra_downsamples.append(
                nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))

        # add NAS FPN connections
        self.fpn_stages = ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

         Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build P6-P7 on top of P5
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return p3, p4, p5, p6, p7
