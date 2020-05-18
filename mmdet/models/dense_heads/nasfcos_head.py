import copy

import torch.nn as nn
from mmcv.cnn import (ConvModule, Scale, bias_init_with_prob,
                      caffe2_xavier_init, normal_init)

from mmdet.models.dense_heads.fcos_head import FCOSHead
from ..builder import HEADS


@HEADS.register_module()
class NASFCOSHead(FCOSHead):
    """Anchor-free head used in `NASFCOS <https://arxiv.org/abs/1906.04423>`_.

    It is quite similar with FCOS head, except for the searched structure
    of classification branch and bbox regression branch, where a structure
    of "dconv3x3, conv3x3, dconv3x3, conv1x1" is utilized instead.

    """

    def _init_layers(self):
        dconv3x3_config = dict(
            type='DCNv2',
            kernel_size=3,
            use_bias=True,
            deformable_groups=2,
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

        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        # retinanet_bias_init
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)

        for branch in [self.cls_convs, self.reg_convs]:
            for module in branch.modules():
                if isinstance(module, ConvModule) \
                        and isinstance(module.conv, nn.Conv2d):
                    caffe2_xavier_init(module.conv)
