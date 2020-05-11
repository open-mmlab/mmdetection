import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, Scale

from mmdet.models.dense_heads.fcos_head import FCOSHead, INF
from mmdet.ops import ModulatedDeformConvPack
from ..builder import HEADS


@HEADS.register_module()
class NASFCOSHead(FCOSHead):
    """Anchor-free head used in `NASFCOS <https://arxiv.org/abs/1906.04423>`_.

    It is quite similar with FCOS head, except for the searched structure
    of classification branch and bbox regression branch, where a structure
    of "dconv3x3, conv3x3, dconv3x3, conv1x1"  is utilized instead.

    Example:
        >>> self = NASFCOSHead(11, 7, ["dconv3x3", "skip", "conv3x3", "skip", "dconv3x3", "conv1x1"])
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 arch,
                 **kwargs):
        self.arch = arch
        super(NASFCOSHead, self).__init__(num_classes,
                                          in_channels,
                                          **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i, op in enumerate(self.arch):
            chn = self.in_channels if i == 0 else self.feat_channels
            if op == "skip" : continue
            elif op=="dconv3x3" or op=="conv3x3" or "conv1x1":
                ccfg = self.conv_cfg if op[0]=="d" else dict(type='Conv')
                ksize = 3 if "3" in op else 1
                use_bias = op=="dconv3x3"
                padding = 1 if op!="conv1x1" else 0
                module = ConvModule(chn, self.feat_channels, ksize,
                                    stride=1, padding=padding,
                                    norm_cfg=self.norm_cfg,
                                    bias=use_bias, conv_cfg=ccfg)
            else:
                raise NotImplementedError

            self.cls_convs.append(copy.deepcopy(module))
            self.reg_convs.append(copy.deepcopy(module))

        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        # retinanet_bias_init
        bias_cls = bias_init_with_prob(0.01)
        for modules in [self.fcos_cls, self.fcos_reg, self.fcos_centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        torch.nn.init.constant_(self.fcos_cls.bias, bias_cls)

        for branch in [self.cls_convs, self.reg_convs]:
            for m in branch:
                for k in m.modules():
                    if hasattr(k, "reset_parameters"):
                        k.reset_parameters()
                if hasattr(m, "conv") and \
                    isinstance(m.conv, ModulatedDeformConvPack):
                    m.conv.init_offset()
                    m.conv.reset_parameters()

