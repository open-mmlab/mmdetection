import numpy
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule,auto_fp16
import torch
from torch import nn
import torch.nn.functional as F

from .fpn import FPN
from ..builder import NECKS
from ..utils import InvertedResidual


def downsample_layer():
    # 原本的设置是3，2,1，这种设置可能是有问题的，还是2,2,0这种方式应该才是正确的……
    return nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
    # return nn.MaxPool2d(kernel_size=2,stride=2)

class WeightedSum(nn.Module):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self):
        super(WeightedSum, self).__init__()

        self.a = nn.Parameter(
            data=torch.ones(size=(4,),dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, model_outputs):
        return self.a[0] * model_outputs[0] + self.a[1] * model_outputs[1] + self.a[2] * model_outputs[2] + self.a[3] * model_outputs[3]

    def compute_output_shape(self, input_shape):
        return input_shape[0]




@NECKS.register_module("RFCR_FPN")
class RFCR_FPN(FPN):
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
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(RFCR_FPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg
        )
        # 每个阶层的1x1下采样的通道数
        self.down_channels = 48
        # 中间的MBConv的输出通道数
        self.extra_channels = 96
        self.rfcr_pConv1 = nn.Conv2d(in_channels[0], self.down_channels, kernel_size=1, padding=0, bias=False)
        self.rfcr_pConv2 = nn.Conv2d(in_channels[1], self.down_channels, kernel_size=1, padding=0, bias=False)
        self.rfcr_pConv3 = nn.Conv2d(in_channels[2], self.down_channels, kernel_size=1, padding=0, bias=False)
        self.rfcr_pConv4 = nn.Conv2d(in_channels[3], self.down_channels, kernel_size=1, padding=0, bias=False)



        self.weighted_sum = WeightedSum()
        # self.MBConv5_5 = InvertedResidual(in_channels=self.down_channels,
        #                                   out_channels=self.extra_channels,
        #                                   with_expand_conv=False,
        #                                   mid_channels=self.down_channels,
        #                                   kernel_size=5,
        #                                   stride=1,
        #                                   act_cfg=dict(type='ReLU6'))

        self.downsample = downsample_layer()
        self.downsample_double = nn.Sequential(downsample_layer(),downsample_layer())


        # 原FPN的init，重写出来是因为经过RFCR模块，出来的特征的通道数变了，所以原本的lateral_conv的通道数也得变掉
        assert isinstance(in_channels, list)
        # 这里改掉了，增加了因为rfcr模块额外增加的通道数
        # self.in_channels = [channel+self.extra_channels for channel in in_channels]
        self.in_channels = [channel+self.down_channels for channel in in_channels]

        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

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


        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                self.in_channels[i],
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

    def RFCR_NECK(self,input):

        b1c = input[0]
        b2c = input[1]
        b3c = input[2]
        b4c = input[3]

        b1c = self.rfcr_pConv1(b1c)
        b2c = self.rfcr_pConv2(b2c)
        b3c = self.rfcr_pConv3(b3c)
        b4c = self.rfcr_pConv4(b4c)

        bc = self.weighted_sum([self.downsample_double(b1c),
                                self.downsample(b2c),
                                b3c,
                                F.interpolate(b4c,scale_factor=2)])

        # bc = self.MBConv5_5(bc)

        b1 = torch.cat([input[0], F.interpolate(F.interpolate(bc, scale_factor=2,mode='bilinear'), scale_factor=2,mode='bilinear')],dim=1)
        b2 = torch.cat([input[1], F.interpolate(bc, scale_factor=2,mode='bilinear')],dim=1)
        b3 = torch.cat([input[2], bc],dim=1)
        b4 = torch.cat([input[3], self.downsample(bc)],dim=1)

        # b1 = torch.add(input[0], F.interpolate(F.interpolate(bc, scale_factor=2), scale_factor=2))
        # b2 = torch.add(input[1], F.interpolate(bc, scale_factor=2))
        # b3 = torch.add(input[2], bc)
        # b4 = torch.add(input[3], self.downsample(bc))

        return b1, b2, b3, b4

    @auto_fp16()
    def forward(self, inputs):

        inputs = self.RFCR_NECK(inputs)

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
        return tuple(outs)