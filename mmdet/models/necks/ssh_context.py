# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class SSHContext(BaseModule):

    def __init__(self,
                 in_channel,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        """Implementation of `SSH context module in.

        <https://arxiv.org/abs/1708.03979>`_.

        Args:
            in_channel (int): Number of input channels used at each scale.
            out_channel (int): Number of output channels used at each scale.
            conv_cfg (dict): Config dict for convolution layer.
                Default: None.
            norm_cfg (dict): Config dict for normalization layer.
                Default: None.
            init_cfg (dict or list[dict], optional): Initialization config
                dict.
        """
        super(SSHContext, self).__init__(init_cfg)
        assert out_channel % 4 == 0

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv3X3 = ConvModule(
            self.in_channel,
            self.out_channel // 2,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv5X5_1 = ConvModule(
            self.in_channel,
            self.out_channel // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

        self.conv5X5_2 = ConvModule(
            self.out_channel // 4,
            self.out_channel // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv7X7_2 = ConvModule(
            self.out_channel // 4,
            self.out_channel // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

        self.conv7X7_3 = ConvModule(
            self.out_channel // 4,
            self.out_channel // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, inputs):
        outs = []
        for x in inputs:
            conv3X3 = self.conv3X3(x)

            conv5X5_1 = self.conv5X5_1(x)
            conv5X5 = self.conv5X5_2(conv5X5_1)

            conv7X7_2 = self.conv7X7_2(conv5X5_1)
            conv7X7 = self.conv7X7_3(conv7X7_2)

            out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
            out = F.relu(out)
            outs.append(out)

        return tuple(outs)
