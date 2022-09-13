# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import NECKS
from mmdet.models.necks.yolox_pafpn import YOLOXPAFPN


class ASFF(nn.Module):
    """ASFF used in `YOLOX-PAI <https://arxiv.org/abs/2208.13040>`_.

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
        self.level = level
        if level == 0:
            self.inter_dim = in_channels[2]
            self.stride_level_1 = ConvModule(
                in_channels[1],
                self.inter_dim,
                3,
                2,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)

            self.stride_level_2 = ConvModule(
                in_channels[0],
                self.inter_dim,
                3,
                2,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)

        elif level == 1:
            self.inter_dim = in_channels[1]
            self.compress_level_0 = ConvModule(
                in_channels[2],
                self.inter_dim,
                1,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)
            self.stride_level_2 = ConvModule(
                in_channels[0],
                self.inter_dim,
                3,
                2,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)

        elif level == 2:
            self.inter_dim = in_channels[0]
            self.compress_level_0 = ConvModule(
                in_channels[2],
                self.inter_dim,
                1,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)
            self.compress_level_1 = ConvModule(
                in_channels[1],
                self.inter_dim,
                1,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)
        else:
            raise ValueError('Invalid level {}'.format(level))

        # add expand layer
        self.expand = ConvModule(
            self.inter_dim,
            self.inter_dim,
            expand_kernel,
            1,
            expand_kernel // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False)

        self.weight_level_0 = ConvModule(
            self.inter_dim,
            asff_channel,
            1,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False)
        self.weight_level_1 = ConvModule(
            self.inter_dim,
            asff_channel,
            1,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False)
        self.weight_level_2 = ConvModule(
            self.inter_dim,
            asff_channel,
            1,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False)

        self.weight_levels = ConvModule(
            asff_channel * 3,
            3,
            1,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False)

    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
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
        return x

    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/4,h*2,w*2]
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2

    def forward(self, x):
        x_level_0 = x[2]
        x_level_1 = x[1]
        x_level_2 = x[0]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = (level_0_resized * levels_weight[:, 0:1, :, :]) + (
            level_1_resized * levels_weight[:, 1:2, :, :]) + (
                level_2_resized * levels_weight[:, 2:, :, :])
        out = self.expand(fused_out_reduced)

        return out


@NECKS.register_module()
class YOLOXASFFPAFPN(YOLOXPAFPN):
    """Path Aggregation Network used in `YOLOX-PAI.

    <https://arxiv.org/abs/2208.13040>`_.

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
        assert len(self.in_channels) == 3,\
            'len(in_channels) should be set to 3.'
        # todo: handle len(self.in_channels) > 3

        self.asff_1 = ASFF(
            self.in_channels,
            level=0,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.asff_2 = ASFF(
            self.in_channels,
            level=1,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.asff_3 = ASFF(
            self.in_channels,
            level=2,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # asff
        outs = tuple(outs)
        pan_out0 = self.asff_1(outs)
        pan_out1 = self.asff_2(outs)
        pan_out2 = self.asff_3(outs)
        outs = [pan_out2, pan_out1, pan_out0]

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
