# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import auto_fp16
import torch

from ..builder import NECKS
from .fpn import FPN
# from mmdet.models.utils.global_attention_carafe import Attention_CARAFE
from ..utils import CARAFE


class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module()
class Attention_PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(Attention_PAFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        self.channel_atts = nn.ModuleList()
        self.spatial_atts = nn.ModuleList()
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        self.upsample_carafes = nn.ModuleList()
        att_chn = 2 * out_channels
        for i in range(self.start_level + 1, self.backbone_end_level):
            # upsample_carafe = Attention_CARAFE(op='upsample',in_channel=out_channels)
            upsample_carafe = CARAFE(c=out_channels, op='upsample', c_mid=64)
            # d_conv = ConvModule(
            #     out_channels,
            #     out_channels,
            #     3,
            #     stride=2,
            #     padding=1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            #     inplace=False)
            # d_conv = Attention_CARAFE(op='downsample',in_channel=out_channels)
            d_conv = CARAFE(c=out_channels, op='downsample', c_mid=16)
            self.spatial_atts.append(nn.Sequential(nn.Conv2d(att_chn, att_chn,
                                                             kernel_size=3, stride=1, padding=1),
                                                   ASPP(att_chn, att_chn // 4),
                                                   ConvModule(att_chn, 1,
                                                              kernel_size=3, stride=1, padding=1,
                                                              norm_cfg=dict(type='BN'),
                                                              act_cfg=dict(type='ReLU')),
                                                   nn.Sigmoid()
                                                   ))
            self.channel_atts.append(nn.Sequential(
                ConvModule(in_channels=att_chn,
                           out_channels=att_chn,
                           kernel_size=1,
                           norm_cfg=dict(type='BN'),
                           act_cfg=dict(type='ReLU')),
                nn.Conv2d(in_channels=att_chn,
                           out_channels=att_chn,
                           kernel_size=1),
                nn.Sigmoid()
            ))
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.upsample_carafes.append(upsample_carafe)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], size=prev_shape, mode='nearest')
            lower_identity = laterals[i].clone()
            # upper_identity = laterals[i - 1].clone()

            lower = F.interpolate(laterals[i], scale_factor=2, mode='bilinear')

            mid = torch.cat((lower, laterals[i - 1]), dim=1)  # 1 * 2c * h * w
            att = self.spatial_atts[i - 1](mid)
            mid = att * mid

            mid = 2 * self.channel_atts[i - 1](mid)
            lower_att, upper_att = torch.split(mid, self.out_channels, dim=1)

            lower_identity = self.upsample_carafes[i - 1](lower_identity)
            lower_result = lower_identity * lower_att
            upper_result = laterals[i - 1] * upper_att
            laterals[i - 1] = lower_result + upper_result

            # 改成复现的GACARAFE看看
            #     laterals[i - 1] = self.upsample_carafes[i - 1](laterals[i],laterals[i - 1])

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            # inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
            # lower_identity = inter_outs[i + 1].clone()
            upper_identity = inter_outs[i].clone()

            upper = F.max_pool2d(inter_outs[i], kernel_size=2, stride=2)

            mid = torch.cat((inter_outs[i + 1], upper), dim=1)  # 1 * 2c * h * w
            att = self.spatial_atts[i](mid)
            mid = att * mid

            mid = 2 * self.channel_atts[i](mid)
            lower_att, upper_att = torch.split(mid, self.out_channels, dim=1)

            upper_identity = self.downsample_convs[i](upper_identity)
            lower_result = inter_outs[i + 1] * lower_att
            upper_result = upper_identity * upper_att
            inter_outs[i + 1] = lower_result + upper_result

            # 同改成复现的GACARAFE
            # inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i+1],inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


def get_module():
    return Attention_PAFPN(in_channels=[256, 512, 1024, 2048],
                           out_channels=256,
                           num_outs=5)
