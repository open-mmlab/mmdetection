import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..plugins import NonLocalBlock2D
from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class BFP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 refine_level=2,
                 refine_type=None,
                 activation=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

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

        self.refine_level = refine_level
        self.refine_type = refine_type

        self.ops = []
        self.rops = []
        for i in range(self.start_level, self.backbone_end_level):
            if i < self.refine_level:
                self.ops.append(F.adaptive_max_pool2d)
                self.rops.append(F.interpolate)
            else:
                self.ops.append(F.interpolate)
                self.rops.append(F.adaptive_max_pool2d)

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=activation)
        elif self.refine_type == 'non_local':
            self.refine = NonLocalBlock2D(
                out_channels,
                reduction=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=activation)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.extra_convs.append(extra_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        ops_params = [
            dict(output_size=inputs[self.refine_level].size()[2:])
            if i < self.refine_level else dict(
                size=inputs[self.refine_level].size()[2:], mode='nearest')
            for i in range(self.start_level, self.backbone_end_level)
        ]
        rops_params = [
            dict(size=inputs[i].size()[2:], mode='nearest') if
            i < self.refine_level else dict(output_size=inputs[i].size()[2:])
            for i in range(self.start_level, self.backbone_end_level)
        ]

        feats = [
            self.ops[i](inputs[i + self.start_level], **ops_params[i])
            for i in range(len(self.ops))
        ]

        bsf = sum(feats) / len(feats)
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        outs = [
            self.rops[i](bsf, **rops_params[i]) + inputs[i + self.start_level]
            for i in range(len(self.rops))
        ]

        used_backbone_levels = len(outs)
        # add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.extra_convs[0](orig))
                else:
                    outs.append(self.extra_convs[0](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i - used_backbone_levels](
                            F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i - used_backbone_levels](
                            outs[-1]))

        return tuple(outs)
