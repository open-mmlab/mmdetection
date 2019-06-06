import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class RFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 conv_cfg=None,
                 normalize=None,
                 activation=None):
        super(RFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

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
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pre_fpn_convs = nn.ModuleList()
        self.rev_lateral_convs = nn.ModuleList()
        self.rev_fpn_convs = nn.ModuleList()
        # levle start from 0,...
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                1,
                1,
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            pre_fpn_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            rev_l_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                1,
                1,
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            rev_fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)


            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.pre_fpn_convs.append(pre_fpn_conv)
            self.rev_lateral_convs.append(rev_l_conv)
            self.rev_fpn_convs.append(rev_fpn_conv)
        self.top_down_deconv = nn.ModuleList()
        self.bottom_up_conv = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            self.top_down_deconv.append(nn.ConvTranspose2d(out_channels,out_channels,4,2,padding=1))
            bu_conv = ConvModule(#  relu+3x3 s1
                out_channels,
                out_channels,
                3,
                2,
                1,# 这里下采样 必须padding
                conv_cfg=conv_cfg,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False,activate_last=True)
            self.bottom_up_conv.append(bu_conv)



        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]# 最后stage的通道数
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        first_outs = inputs
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        #for i in range(used_backbone_levels - 1, 0, -1):
        #    laterals[i - 1] += F.interpolate(
        #        laterals[i], scale_factor=2, mode='nearest')

        for i in range(used_backbone_levels - 1, 0, -1):
            #print(laterals[i-1].shape)
            #print(laterals[i].shape)
            #print(self.top_down_deconv[i-1](laterals[i]).shape)
            laterals[i-1] += self.top_down_deconv[i-1](laterals[i])
        second_outs = laterals
        #build rev_laterals

        rev_laterals = [rev_lateral_conv(laterals[ii])
                    for ii, rev_lateral_conv in enumerate(self.rev_lateral_convs)]

        # self.bottom_up_conv
        for i in range(0, used_backbone_levels - 1):
            #print(self.bottom_up_conv[i](rev_laterals[i]).shape)
            #print(rev_laterals[i+1].shape)
            rev_laterals[i+1] += self.bottom_up_conv[i](rev_laterals[i])
        third_outs = rev_laterals
        # build outputs
        #Q1
        """
        tem = [inputs[ii] for ii in range(self.start_level,self.backbone_end_level)]
        outs_q1 = [self.pre_fpn_convs[i](tem[i]) for i in range(used_backbone_levels)]
        #Q2
        # part 1: from original levels
        outs_q2 = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        #Q3

        outs_q3 = [
            self.rev_fpn_convs[i](rev_laterals[i]) for i in range(used_backbone_levels)
        ]
        """
        #print(len(outs_q3))
        # part 2: add extra levels
        if self.num_outs > len(outs_q3):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs_q3.append(F.max_pool2d(outs_q3[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]#the last feature of inputs
                    outs_q3.append(self.fpn_convs[used_backbone_levels](orig))# extra layer on final feature
                else:
                    outs_q3.append(self.fpn_convs[used_backbone_levels](outs_q3[-1]))#extra layer on final out
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs_q3.append(self.fpn_convs[i](outs_q3[-1]))
        #print(len(outs_q3))
        # part 2: add extra levels
        if self.num_outs > len(outs_q2):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs_q2.append(F.max_pool2d(outs_q2[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]#the last feature of inputs
                    outs_q2.append(self.fpn_convs[used_backbone_levels](orig))# extra layer on final feature
                else:
                    outs_q2.append(self.fpn_convs[used_backbone_levels](outs_q2[-1]))#extra layer on final out
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs_q2.append(self.fpn_convs[i](outs_q2[-1]))
        # part 2: add extra levels
        if self.num_outs > len(outs_q1):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs_q1.append(F.max_pool2d(outs_q1[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]#the last feature of inputs
                    outs_q1.append(self.fpn_convs[used_backbone_levels](orig))# extra layer on final feature
                else:
                    outs_q1.append(self.fpn_convs[used_backbone_levels](outs_q1[-1]))#extra layer on final out
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs_q1.append(self.fpn_convs[i](outs_q1[-1]))

        #assert len(outs_q1) == len(outs_q2) == len(outs_q3) == self.num_outs
        #return tuple(outs_q1), tuple(outs_q2), tuple(outs_q3)
        assert len(first_outs) == len(second_outs) == len(third_outs)
        return tuple(first_outs), tuple(second_outs), tuple(third_outs)
