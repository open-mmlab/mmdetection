# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmdet.models.utils.lka_layer import AttentionModule

from mmdet.models.utils.carafe import CARAFE

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class PAFPN_LKAATTENTION_UNIFIED_CARAFE(FPN):
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
        super(PAFPN_LKAATTENTION_UNIFIED_CARAFE, self).__init__(
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
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        self.fpn_att_convs = nn.ModuleList()
        self.pafpn_att_convs = nn.ModuleList()
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        for i in range(self.start_level, self.backbone_end_level):
            # fpn_att_conv = AttentionModule(dim=out_channels,
            #                                norm_cfg = norm_cfg,
            #                                act_cfg = act_cfg)
            # pafpn_att_conv = AttentionModule(dim=out_channels)
            fpn_att_conv = nn.Sequential(AttentionModule(dim=out_channels,
                                           norm_cfg=norm_cfg,
                                           act_cfg=act_cfg),
                                         ConvModule(in_channels=out_channels, out_channels=out_channels,
                                                    kernel_size=3, padding=1,
                                                    act_cfg=dict(type='Sigmoid'))
                                         )
            pafpn_att_conv = nn.Sequential(AttentionModule(dim=out_channels,
                                           norm_cfg=norm_cfg,
                                           act_cfg=act_cfg),
                                         ConvModule(in_channels=out_channels, out_channels=out_channels,
                                                    kernel_size=3, padding=1,
                                                    act_cfg=dict(type='Sigmoid'))
                                         )
            self.fpn_att_convs.append(fpn_att_conv)
            self.pafpn_att_convs.append(pafpn_att_conv)

        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = CARAFE(out_channels,
                            op='downsample',
                            scale=2,
                            c_mid=16)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.used_backbone_levels = len(self.lateral_convs)
        self.upsample_modules = nn.ModuleList()
        self.extra_downsample_modules = nn.ModuleList()

        for i in range(self.used_backbone_levels - 1, 0, -1):
            self.upsample_modules.append(CARAFE(out_channels,
                                                op='upsample',
                                                scale=2,
                                                c_mid=64))

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # lzj 应用注意力
        fpn_att_list = [self.fpn_att_convs[i](laterals[i]) for i in range(len(laterals))]
        laterals = [(1 + fpn_att_list[i]) * laterals[i] for i in range(len(laterals))]

        # build top-down path

        for i in range(self.used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], size=prev_shape, mode='nearest')

            laterals[i - 1] += self.upsample_modules[i - 1](laterals[i]) * F.interpolate(
                fpn_att_list[i], scale_factor=2, mode='bilinear')
            # laterals[i - 1] += self.upsample_modules[i - 1](laterals[i]) * fpn_att_list[i - 1]

        # build outputs
        # part 1: from original levels
        # inter_outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(self.used_backbone_levels)
        # ]

        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.used_backbone_levels)
        ]
        # 为pafpn的中间输出也添加注意力
        inter_att_list = [self.pafpn_att_convs[i](inter_outs[i]) for i in range(len(laterals))]
        inter_outs = [(1 + inter_att_list[i]) * inter_outs[i] for i in range(len(laterals))]

        # part 2: add bottom-up path
        # 添加注意力
        for i in range(0, self.used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i]) * F.max_pool2d(
                inter_att_list[i], kernel_size=2,stride=2)

            # inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i]) * inter_att_list[i + 1]
            # inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i]) * fpn_att_list[i + 1]

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, self.used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - self.used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[self.used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[self.used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[self.used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(self.used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    tensor = [torch.rand(size=(1, 256, 128, 160)),
              torch.rand(size=(1, 512, 64, 80)),
              torch.rand(size=(1, 1024, 32, 40)),
              torch.rand(size=(1, 2048, 16, 20))]

    model = PAFPN_LKAATTENTION_UNIFIED_CARAFE(in_channels=[256, 512, 1024, 2048],
                                              out_channels=256,
                                              num_outs=5)

    results = model(tensor)
    for result in results:
        print(result.shape)
