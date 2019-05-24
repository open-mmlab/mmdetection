import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..plugins import NonLocalBlock2D
from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Args:
        in_channels (list/int): number of channels for each branch.
        num_levels (int): number of input branches.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        refine_level (int): index of corresponding level of BSF.
        refine_type (str): type of method for refining features,
            currently support [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None,
                 refine_level=2,
                 refine_type=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        if isinstance(in_channels, list):
            self.channels = in_channels[0]
            assert len(set(in_channels)) == 1
        elif isinstance(in_channels, int):
            self.channels = in_channels
        else:
            raise TypeError(
                'The in_channels should be int or list but found {}.'.format(
                    type(in_channels)))
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type

        self.gather_ops = []
        self.scatter_rops = []
        for i in range(self.num_levels):
            if i < self.refine_level:
                self.gather_ops.append(F.adaptive_max_pool2d)
                self.scatter_rops.append(F.interpolate)
            else:
                self.gather_ops.append(F.interpolate)
                self.scatter_rops.append(F.adaptive_max_pool2d)

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocalBlock2D(
                self.channels,
                reduction=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        gather_params = []
        scatter_params = []
        for i in range(self.num_levels):
            if i < self.refine_level:
                gather_params.append(
                    dict(output_size=inputs[self.refine_level].size()[2:]))
                scatter_params.append(
                    dict(size=inputs[i].size()[2:], mode='nearest'))
            else:
                gather_params.append(
                    dict(
                        size=inputs[self.refine_level].size()[2:],
                        mode='nearest'))
                scatter_params.append(dict(output_size=inputs[i].size()[2:]))

        feats = [
            self.gather_ops[i](inputs[i], **gather_params[i])
            for i in range(self.num_levels)
        ]

        bsf = sum(feats) / len(feats)
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        outs = [
            self.scatter_rops[i](bsf, **scatter_params[i]) + inputs[i]
            for i in range(self.num_levels)
        ]

        return tuple(outs)
