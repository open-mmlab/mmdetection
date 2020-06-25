import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init

from mmdet.ops.merge_cells import ConcatCell
from ..builder import NECKS


@NECKS.register_module()
class NASFCOS_FPN(nn.Module):
    """FPN structure in NASFPN

    Implementation of paper `NAS-FCOS: Fast Neural Architecture Search for
    Object Detection <https://arxiv.org/abs/1906.04423>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): It decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None):
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
            adapt_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.adapt_convs.append(adapt_conv)

        # C2 is omitted according to the paper
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        def build_concat_cell(with_input1_conv, with_input2_conv):
            cell_conv_cfg = dict(
                kernel_size=1, padding=0, bias=False, groups=out_channels)
            return ConcatCell(
                in_channels=out_channels,
                out_channels=out_channels,
                with_out_conv=True,
                out_conv_cfg=cell_conv_cfg,
                out_norm_cfg=dict(type='BN'),
                out_conv_order=('norm', 'act', 'conv'),
                with_input1_conv=with_input1_conv,
                with_input2_conv=with_input2_conv,
                input_conv_cfg=conv_cfg,
                input_norm_cfg=norm_cfg,
                upsample_mode='nearest')

        # Denote c3=f0, c4=f1, c5=f2 for convince
        self.fpn = nn.ModuleDict()
        self.fpn['c22_1'] = build_concat_cell(True, True)
        self.fpn['c22_2'] = build_concat_cell(True, True)
        self.fpn['c32'] = build_concat_cell(True, False)
        self.fpn['c02'] = build_concat_cell(True, False)
        self.fpn['c42'] = build_concat_cell(True, True)
        self.fpn['c36'] = build_concat_cell(True, True)
        self.fpn['c61'] = build_concat_cell(True, True)  # f9
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_act_cfg = None if i == 0 \
                else dict(type='ReLU', inplace=False)
            self.extra_downsamples.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=extra_act_cfg,
                    order=('act', 'norm', 'conv')))

    def forward(self, inputs):
        """Forward function"""
        feats = [
            adapt_conv(inputs[i + self.start_level])
            for i, adapt_conv in enumerate(self.adapt_convs)
        ]

        for (i, module_name) in enumerate(self.fpn):
            idx_1, idx_2 = int(module_name[1]), int(module_name[2])
            res = self.fpn[module_name](feats[idx_1], feats[idx_2])
            feats.append(res)

        ret = []
        for (idx, input_idx) in zip([9, 8, 7], [1, 2, 3]):  # add P3, P4, P5
            feats1, feats2 = feats[idx], feats[5]
            feats2_resize = F.interpolate(
                feats2,
                size=feats1.size()[2:],
                mode='bilinear',
                align_corners=False)

            feats_sum = feats1 + feats2_resize
            ret.append(
                F.interpolate(
                    feats_sum,
                    size=inputs[input_idx].size()[2:],
                    mode='bilinear',
                    align_corners=False))

        for submodule in self.extra_downsamples:
            ret.append(submodule(ret[-1]))

        return tuple(ret)

    def init_weights(self):
        """Initialize the weights of module"""
        for module in self.fpn.values():
            if hasattr(module, 'conv_out'):
                caffe2_xavier_init(module.out_conv.conv)

        for modules in [
                self.adapt_convs.modules(),
                self.extra_downsamples.modules()
        ]:
            for module in modules:
                if isinstance(module, nn.Conv2d):
                    caffe2_xavier_init(module)
