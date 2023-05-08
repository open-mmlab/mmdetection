# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmengine.model import BaseModule, normal_init

from mmdet.registry import MODELS


@MODELS.register_module()
class SemanticFPN(BaseModule):
    """Semantic FPN.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels in the feature map.
        out_channels (int): Number of channels in the output feature map.
        start_level (int): Starting feature map level.
        end_level (int): Ending feature map level.
        output_level (int): Which level of feature map to output.
        positional_encoding_level (int): Which level of feature map to
            add positional encoding to.
        positional_encoding_cfg (:obj:`mmcv.Config` or None): Config of
            positional encoding. Defaults to None.
        add_aux_conv (bool): Whether to add an auxiliary convolution layer.
            Defaults to False.
        act_cfg (:obj:`mmcv.Config`): Config of activation.
            Defaults to dict(type='ReLU', inplace=True).
        out_act_cfg (:obj:`mmcv.Config`): Config of output activation.
            Defaults to dict(type='ReLU').
        conv_cfg (:obj:`mmcv.Config` or None): Config of convolution layer.
            Defaults to None.
        norm_cfg (:obj:`mmcv.Config` or None): Config of normalization layer.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 start_level,
                 end_level,
                 output_level,
                 positional_encoding_level,
                 positional_encoding_cfg=None,
                 add_aux_conv=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert 0 <= start_level and start_level <= output_level \
            and output_level <= end_level
        assert start_level <= positional_encoding_level \
            and positional_encoding_level <= end_level

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        self.start_level = start_level
        self.end_level = end_level
        self.output_level = output_level
        self.positional_encoding_level = positional_encoding_level

        self.positional_encoding_cfg = positional_encoding_cfg
        self.add_aux_conv = add_aux_conv
        self.act_cfg = act_cfg
        self.out_act_cfg = out_act_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layer()

    def _init_layer(self):
        if self.positional_encoding_cfg:
            self.positional_encoding = build_positional_encoding(
                self.positional_encoding_cfg)
        else:
            self.positional_encoding = None
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()

            if i < self.output_level:
                # downsample
                for j in range(self.output_level - i):
                    downsampled_conv = ConvModule(
                        self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j),
                                               downsampled_conv)
            elif i == self.output_level:
                # just conv
                one_conv = ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(i - self.output_level),
                                           one_conv)
            else:
                # upsample
                for j in range(i):
                    one_conv = ConvModule(
                        self.in_channels if j == 0 else self.feat_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    if j < i - self.output_level:
                        one_upsample = nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
                        convs_per_level.add_module('upsample' + str(j),
                                                   one_upsample)
            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            self.feat_channels,
            self.out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)

        if self.add_aux_conv:
            self.aux_conv = ConvModule(
                self.feat_channels,
                self.out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg)
        else:
            self.aux_conv = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, feats):
        """
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Union[tuple[Tensor, Tensor], tuple[Tensor, None]]: Fused feature
            maps. The second tensor is None when ``add_aux_conv`` is False.
        """
        multi_level_feats = []
        for i in range(self.start_level, self.end_level + 1):
            feat = feats[i]
            if (i == self.positional_encoding_level
                    and self.positional_encoding is not None):
                ignore_mask = feat.new_zeros(
                    (feat.shape[0], ) + feat.shape[-2:], dtype=torch.bool)
                positional_encoding = self.positional_encoding(ignore_mask)
                feat = feat + positional_encoding

            convs_per_level = self.convs_all_levels[i]
            feat = convs_per_level(feat)
            multi_level_feats.append(feat)

        feats_fused = sum(multi_level_feats)
        out = self.conv_pred(feats_fused)

        if self.aux_conv is not None:
            aux_out = self.aux_conv(feats_fused)
            return out, aux_out
        else:
            return out, None
