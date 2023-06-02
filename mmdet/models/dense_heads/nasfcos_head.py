# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from mmcv.cnn import ConvModule, Scale

from mmdet.models.dense_heads.fcos_head import FCOSHead
from ..builder import HEADS


@HEADS.register_module()
class NASFCOSHead(FCOSHead):
    """Anchor-free head used in `NASFCOS <https://arxiv.org/abs/1906.04423>`_.

    It is quite similar with FCOS head, except for the searched structure of
    classification branch and bbox regression branch, where a structure of
    "dconv3x3, conv3x3, dconv3x3, conv1x1" is utilized instead.
    """

    def __init__(self, *args, init_cfg=None, **kwargs):
        if init_cfg is None:
            init_cfg = [
                dict(type='Caffe2Xavier', layer=['ConvModule', 'Conv2d']),
                dict(
                    type='Normal',
                    std=0.01,
                    override=[
                        dict(name='conv_reg'),
                        dict(name='conv_centerness'),
                        dict(
                            name='conv_cls',
                            type='Normal',
                            std=0.01,
                            bias_prob=0.01)
                    ]),
            ]
        super(NASFCOSHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        dconv3x3_config = dict(
            type='DCNv2',
            kernel_size=3,
            use_bias=True,
            deform_groups=2,
            padding=1)
        conv3x3_config = dict(type='Conv', kernel_size=3, padding=1)
        conv1x1_config = dict(type='Conv', kernel_size=1)

        self.arch_config = [
            dconv3x3_config, conv3x3_config, dconv3x3_config, conv1x1_config
        ]
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i, op_ in enumerate(self.arch_config):
            op = copy.deepcopy(op_)
            chn = self.in_channels if i == 0 else self.feat_channels
            assert isinstance(op, dict)
            use_bias = op.pop('use_bias', False)
            padding = op.pop('padding', 0)
            kernel_size = op.pop('kernel_size')
            module = ConvModule(
                chn,
                self.feat_channels,
                kernel_size,
                stride=1,
                padding=padding,
                norm_cfg=self.norm_cfg,
                bias=use_bias,
                conv_cfg=op)

            self.cls_convs.append(copy.deepcopy(module))
            self.reg_convs.append(copy.deepcopy(module))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
