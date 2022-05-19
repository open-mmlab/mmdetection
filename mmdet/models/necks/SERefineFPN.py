# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from ..utils import SELayer


@NECKS.register_module()
class SERefineFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

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
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

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
                 target_stage,
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
        super(SERefineFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        # add by lzj
        # the stage to adapt to that size
        self.target_stage = target_stage

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # add by lzj
        self.adapt_same_size_modules = []
        self.adapt_origin_size_modules = []

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

        # add extra conv layers (e.g., RetinaNet)
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

        # self.ChannelShuffle = nn.ChannelShuffle(1)

        consistency_adjust_kernel = 5
        self.consistency_adjust = ConvModule(out_channels * num_outs,
                                             out_channels * num_outs,
                                             kernel_size=consistency_adjust_kernel,
                                             padding=(consistency_adjust_kernel - 1) // 2,
                                             conv_cfg=dict(type='DCNv2'),
                                             norm_cfg=None,
                                             act_cfg=None)

        self.se_layers = nn.ModuleList()
        for i in range(0, num_outs):
            self.se_layers.append(SELayer(channels=out_channels))

        self.adapt_to_same_size_init(target_stage=target_stage)
        self.adapt_to_origin_size_init(target_stage=target_stage)

    def adapt_to_same_size_init(self, target_stage):
        assert target_stage in range(0, self.num_outs)

        for i in range(0, self.num_outs):
            delta = abs(i - target_stage)
            times = pow(2, delta)
            if i < target_stage:
                same_downsamples = nn.Sequential()
                for i in range(0, delta):
                    same_downsamples.add_module(name='extra_same_size_'+str(i),module=nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
                self.adapt_same_size_modules.append(same_downsamples)
            elif i == target_stage:
                self.adapt_same_size_modules.append('pass')
            else:
                self.adapt_same_size_modules.append(nn.UpsamplingBilinear2d(scale_factor=times))

    def adapt_to_origin_size_init(self, target_stage):
        assert target_stage in range(0, self.num_outs)
        for i in range(0, self.num_outs):
            delta = abs(target_stage - i)
            times = pow(2, delta)
            if i < target_stage:
                self.adapt_origin_size_modules.append(nn.UpsamplingBilinear2d(scale_factor=times))
                # inputs[i] = F.interpolate(inputs[i], target_size, mode='bilinear')
            elif i == target_stage:
                self.adapt_origin_size_modules.append('pass')
            else:
                origin_downsamples = nn.Sequential()
                for i in range(0, delta):
                    origin_downsamples.add_module(name='extra_origin_size_'+str(i),module=nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
                self.adapt_origin_size_modules.append(origin_downsamples)
                # target_size = ((int)(start_size[2] * times), (int)(start_size[3] * times))
                # inputs[i] = nn.AdaptiveMaxPool2d(target_size).cuda()(inputs[i])

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
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # add by lzj feature fusion
        # adapt to same size
        # outs = self.adapt_to_same_size(outs, target_stage=self.target_stage)
        # print('before:')
        # self.print_list_tensor_shape(outs)
        outs = [
            self.adapt_same_size_modules[i](outs[i]) if self.adapt_same_size_modules[i] != 'pass' else outs[i] \
            for i in range(self.num_outs)
        ]
        # print('after:')
        # self.print_list_tensor_shape(outs)
        concentrates = outs[0]
        for i in range(1, len(outs)):
            concentrates = torch.cat([concentrates, outs[i]], dim=1)

        # pytorch ChannelShuffle don't support CUDA platform so use ShuffleNetV1's implemention
        # concentrates = self.ChannelShuffle(concentrates)
        concentrates = self.channel_shuffle(concentrates, group=1)
        concentrates = self.consistency_adjust(concentrates)

        # the 3rd of the shape is channels,divide
        concentrates = list(torch.chunk(concentrates, self.num_outs, dim=1))

        for i in range(0, len(concentrates)):
            concentrates[i] = self.se_layers[i](concentrates[i])

        concentrates = [
            self.adapt_origin_size_modules[i](concentrates[i]) \
                if self.adapt_origin_size_modules[i] != 'pass' \
                else concentrates[i] \
            for i in range(self.num_outs)
        ]
        # concentrates = self.adapt_to_origin_size(concentrates, target_stage=self.target_stage)

        # print('after origin size:')
        # self.print_list_tensor_shape(concentrates)

        return tuple(concentrates)

    def print_list_tensor_shape(self, inputs):
        assert isinstance(inputs, list) or isinstance(inputs, tuple)
        for input in inputs:
            print(input.shape)

    def channel_shuffle(self, x, group):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x
