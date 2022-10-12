# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import NECKS
from mmdet.models.necks.yolox_pafpn import YOLOXPAFPN


class ASFF(nn.Module):
    """Adaptively Spatial Feature Fusion (ASFF) module.

    This implementation is based on `ASFF <https://arxiv.org/abs/1911.09516>`_
    used in `YOLOX-PAI <https://arxiv.org/abs/2208.13040>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        level (int): The level of the input feature.
        asff_channel (int): The hidden channel of the attention layer in
            ASFF. Default: 2.
        expand_kernel (int): Expand kernel size of the expand layer.
            Default: 3.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='SiLU')
    """

    def __init__(self,
                 in_channels,
                 level,
                 asff_channel=2,
                 expand_kernel=3,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU')):
        super(ASFF, self).__init__()
        self.in_channels = in_channels
        self.level = level
        self.mlvl_feat_convs = nn.ModuleList()
        self.mlvl_importance_convs = nn.ModuleList()
        self.inter_dim = in_channels[level]
        for i, in_channel in enumerate(in_channels):
            if i == self.level:
                self.mlvl_feat_convs.append(nn.Identity())
            elif i < self.level:
                self.mlvl_feat_convs.append(
                    ConvModule(
                        in_channel,
                        self.inter_dim,
                        3,
                        stride=2,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            elif i > self.level:
                self.mlvl_feat_convs.append(
                    ConvModule(
                        in_channel,
                        self.inter_dim,
                        1,
                        stride=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

            self.mlvl_importance_convs.append(
                ConvModule(
                    self.inter_dim,
                    asff_channel,
                    1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.expand = ConvModule(
            self.inter_dim,
            self.inter_dim,
            expand_kernel,
            stride=1,
            padding=expand_kernel // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_importance_conv = ConvModule(
            asff_channel * len(in_channels),
            len(in_channels),
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        assert len(x) == len(self.in_channels)

        mlvl_feats = []
        mlvl_importance = []  # spatial importance map
        for i, feat in enumerate(x):
            for _ in range(self.level - i - 1):
                feat = F.max_pool2d(feat, 3, stride=2, padding=1)
            feat = self.mlvl_feat_convs[i](feat)
            if i > self.level:
                feat = F.interpolate(
                    feat, scale_factor=2**(i - self.level), mode='nearest')
            mlvl_feats.append(feat)
            mlvl_importance.append(self.mlvl_importance_convs[i](feat))

        mlvl_importance = torch.cat(mlvl_importance, 1)
        mlvl_importance = self.final_importance_conv(mlvl_importance)
        mlvl_importance = F.softmax(mlvl_importance, dim=1)

        fused_out_reduced = torch.sum(
            torch.stack([
                mlvl_feats[i] * mlvl_importance[:, i:i + 1, :, :]
                for i in range(len(x))
            ]),
            dim=0)
        out = self.expand(fused_out_reduced)

        return out


@NECKS.register_module()
class YOLOXASFFPAFPN(YOLOXPAFPN):
    """Path Aggregation Network with ASFF used in YOLOX-PAI.

    See `YOLOX-PAI <https://arxiv.org/abs/2208.13040>`_ for details.

    Args:
        asff_channel (int): The hidden channel of the attention layer in
            ASFF. Default: 2.
        expand_kernel (int): Expand kernel size of the expand layer.
            Default: 3.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='SiLU')
    """

    def __init__(self,
                 *args,
                 asff_channel=2,
                 expand_kernel=3,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 **kwargs):
        super().__init__(*args, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)

        self.asffs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.asffs.append(
                ASFF(
                    self.in_channels,
                    level=i,
                    asff_channel=asff_channel,
                    expand_kernel=expand_kernel,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # asff
        outs = tuple(outs)
        asff_outs = []
        for asff in self.asffs:
            asff_outs.append(asff(outs))

        # out convs
        for idx, conv in enumerate(self.out_convs):
            asff_outs[idx] = conv(asff_outs[idx])

        return tuple(asff_outs)
