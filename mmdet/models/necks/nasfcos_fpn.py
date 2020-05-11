import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..builder import NECKS

from mmdet.ops import ModulatedDeformConvPack
from mmcv.cnn import ConvModule


class MergingCell(nn.Module):
    def __init__(self, x1_op, x2_op, out_conv,
                 channels, conv_cfg=None, norm_cfg=None):
        super(MergingCell, self).__init__()

        self.op1 = self._build_op_block(x1_op, channels, conv_cfg, norm_cfg)
        self.op2 = self._build_op_block(x2_op, channels, conv_cfg, norm_cfg)

        if out_conv:
            self.out_conv = \
                ConvModule(channels*2, channels, 1,
                            padding=0, bias=False,
                            groups=channels,
                            norm_cfg = dict(type='BN', affine=True),
                            act_cfg = dict(type='ReLU', inplace = False),
                            order=('norm', 'act', 'conv'))

    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def _build_op_block(self, op, c, conv_cfg, norm_cfg):
        if op=="conv":
            return ConvModule(c, c, 3, padding=1,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              bias=True)

        elif op=="skip":
            return nn.Sequential()
        else:
            raise NotImplementedError

    def _resize_largest(self, x1, x2):
        # resize both x and y to max_size(x, y)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:],
                             mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:],
                             mode='bilinear')(x1)
        return x1, x2

    def forward(self, x1, x2):
        out1 = self.op1(x1)
        out2 = self.op2(x2)
        out1, out2 = self._resize_largest(out1, out2)
        out_op = self._binary_op(out1, out2)

        if hasattr(self, "out_conv"):
            out = self.out_conv(out_op)

        return out

class SumCell(MergingCell):
    def __init__(self, x1_op, x2_op,
                 channels=None, conv_cfg=None, norm_cfg=None):
        super(SumCell, self).__init__(x1_op, x2_op,
                                      False, channels,
                                      conv_cfg, norm_cfg)

    def _binary_op(self, x1, x2):
        return x1 + x2

class ConcatCell(MergingCell):
    def __init__(self, x1_op, x2_op,
                 channels, conv_cfg=None, norm_cfg=None):
        super(ConcatCell, self).__init__(x1_op, x2_op,
                                         True, channels,
                                         conv_cfg, norm_cfg)

    def _binary_op(self, x1, x2):
        ret = torch.cat([x1, x2], dim=1)
        return ret


@NECKS.register_module
class NASFCOS_FPN(nn.Module):
    """FPN structure in NASFPN

    NAS-FCOS: Fast Neural Architecture Search for Object Detection
    <https://arxiv.org/abs/1906.04423>
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=False,
                 conv_cfg = None,
                 norm_cfg = None):
        super(NASFCOS_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.adapt_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            adapt_conv = ConvModule(in_channels[i], out_channels, 1,
                                    stride=1, padding=0, bias=False,
                                    norm_cfg=dict(type='BN'),
                                    act_cfg=dict(type='ReLU', inplace=False))
            self.adapt_convs.append(adapt_conv)

        # C2 is omitted according to the paper
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        # Donate c3=f0, c4=f1, c5=f2 for convince
        self.fpn = nn.ModuleDict()
        self.fpn["c22_1"] = ConcatCell("conv", "conv", out_channels, conv_cfg, norm_cfg) # f3
        self.fpn["c22_2"] = ConcatCell("conv", "conv", out_channels, conv_cfg, norm_cfg) # f4
        self.fpn["c32"] = ConcatCell("conv", "skip", out_channels, conv_cfg, norm_cfg)  # f5
        self.fpn["c02"] = ConcatCell("conv", "skip", out_channels, conv_cfg, norm_cfg)   # f6
        self.fpn["c42"] = ConcatCell("conv", "conv", out_channels, conv_cfg, norm_cfg)  # f7
        self.fpn["c36"] = ConcatCell("conv", "conv", out_channels, conv_cfg, norm_cfg)   # f8
        self.fpn["c61"] = ConcatCell("conv", "conv", out_channels, conv_cfg, norm_cfg)   # f9
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_act_cfg = None if i == 0 else dict(type='ReLU', inplace=False)
            self.extra_downsamples.append(
                ConvModule(out_channels, out_channels, 3,
                           stride=2, padding=1, norm_cfg=None,
                           act_cfg=extra_act_cfg,
                           order=('act', 'norm', 'conv')))

    def forward(self, inputs):
        feats = [
            adapt_conv(inputs[i + self.start_level])
            for i, adapt_conv in enumerate(self.adapt_convs)
        ]

        for (i, module_name) in enumerate(self.fpn):
            idx_1, idx_2 = int(module_name[1]), int(module_name[2])
            res = self.fpn[module_name](feats[idx_1], feats[idx_2])
            feats.append(res)

        ret = []
        for (idx, input_idx) in zip([9, 8, 7], [1, 2, 3]): # add P3, P4, P5
            feats1, feats2 = feats[idx], feats[5]
            feats2_resize = F.interpolate(feats2, size=feats1.size()[2:],
                                          mode='bilinear',
                                          align_corners=False)

            feats_sum = feats1 + feats2_resize
            ret.append(F.interpolate(feats_sum,
                                     size=inputs[input_idx].size()[2:],
                                     mode='bilinear', align_corners=False))

        for submodule in self.extra_downsamples:
            ret.append(submodule(ret[-1]))

        return tuple(ret)

    def init_weights(self):
        for m in self.adapt_convs.modules():
            if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.BatchNorm2d):
                    m.reset_parameters()

        for m in self.extra_downsamples.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        for k, m in self.fpn.items():
            for op in [m.op1, m.op2]:
                if isinstance(op, ConvModule):
                    if isinstance(op.conv, ModulatedDeformConvPack):
                        op.conv.reset_parameters()
                    op.bn.reset_parameters()
            if hasattr(m, "out_conv"):
                m.out_conv.conv.reset_parameters()
                m.out_conv.bn.reset_parameters()


