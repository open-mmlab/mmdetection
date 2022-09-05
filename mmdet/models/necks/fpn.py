# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): 长度为n,每种尺度的输入通道数.该参数对下面部分参数影响重大!
        out_channels (int): 输出通道数(用于统一每种尺度).
        num_outs (int): FPN网络输出的层级数量.
        start_level (int): 构建FPN时会有n层特征图(index∈[0, n-1])输入,
            但这些特征图并不总是全部都会使用到.该参数控制着从哪个index开始使用特征图.
            默认为0.即全部都使用
        end_level (int): 对应start_level的end_level,左闭右闭. 默认: -1, 意为n-1.
        add_extra_convs (bool | str): 如果是布尔值,是否添加额外的卷积层.
            默认为False.
            为True时, 它等价于 `add_extra_convs='on_input'`.
            如果是str,它指定额外convs的源特征图,不过只允许以下选项

            - 'on_input': 输入neck的最上层特征图 (即in_channels中最上层的特征图).
            - 'on_lateral': 横向转换后的最后一个特征图,即仅进行1x1卷积的最上层特征图.
            - 'on_output': 进行3x3卷积后的最上层卷积.
        relu_before_extra_convs (bool): 是否在额外卷积之前应用relu. 默认为False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): 卷积层的配置字典. Default: None.
        norm_cfg (dict): norm层的配置字典. Default: None.
        act_cfg (dict): 激活层的配置字典. Default: None.
        upsample_cfg (dict): 插值(上采样)层的配置字典. Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): 初始化配置字典.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
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
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins  # 下面使用range所以为左闭右开
            assert num_outs >= self.num_ins - start_level
        else:
            # 如果 end_level 不是最后一个特征图, 则不会产生额外的卷积或特征图
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # 可选的值: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # 为True时的默认值
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()  # 仅包含升降维卷积
        self.fpn_convs = nn.ModuleList()    # 包含升降维后的3x3卷积以及可能的额外卷积

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 额外添加的卷积层数量 (比如. RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # 1x1卷积过程
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 从上到下下采样以及相加过程
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # FPN层的输出结果
        # part 1: 3x3卷积过程
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: 产生额外特征图过程
        if self.num_outs > len(outs):
            # 如果不额外使用卷积的话,使用max pool在输出之上获得更多尺寸的特征图
            # (比如. Faster R-CNN, Mask R-CNN)
            # 注!该种方式的特征图来自3x3卷积的最上层,额外产生多少个特征图就连续池化多少次
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # 使用额外卷积的情况 (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':  # 来自backbone
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':  # 来自1x1卷积
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':  # 来自3x3卷积
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                # 由于不确定特征图来自哪里,所以这里需要单独添加进outs
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                # 一般的模型,比如 RetinaNet ,Faster-RCNN等是不会再产生额外的特征图了
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
