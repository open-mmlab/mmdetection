import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import HEADS
from .base_semantic_head import BaseSemanticHead


class PanFpnSubNet(nn.Module):

    def __init__(self,
                 in_channels,
                 inner_channels,
                 num_layers=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 num_upsample=None):
        super(PanFpnSubNet, self).__init__()
        if num_upsample is None:  # performs 2x upsample after each conv module
            num_upsample = num_layers

        self.upsample_rate = 2
        self.num_layers = num_layers
        self.num_upsample = num_upsample
        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            conv.append(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    inner_channels,
                    kernel_size=3,
                    stride=1,
                    # not using dilated conv
                    padding=1,
                    dilation=1,
                ))
            in_channels = inner_channels
            if norm_cfg is not None:
                norm_name, norm_layer = build_norm_layer(
                    norm_cfg, in_channels, postfix=i)
                conv.append(norm_layer)
            conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.Sequential(*conv))
        self.init_weights()

    def forward(self, x):
        num_upsample = self.num_upsample
        for i in range(self.num_layers):
            x = self.conv[i](x)
            if num_upsample > 0:
                num_upsample -= 1
                x = F.interpolate(
                    x,
                    None,
                    self.upsample_rate,
                    mode='bilinear',
                    align_corners=False)
        return x

    def init_weights(self):  # calling default initialize
        pass


@HEADS.register_module()
class PanopticFpnHead(BaseSemanticHead):

    def __init__(self,
                 num_classes,
                 in_channels=128,
                 inner_channels=128,
                 num_stages=4,
                 num_feats=-1,
                 fg_range=(1, 80),
                 bg_range=(81, 133),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_semantic=dict(type='CrossEntropyLoss', loss_weight=1.0)):
        super(PanopticFpnHead, self).__init__(num_classes, num_feats,
                                              loss_semantic)
        self.fg_range = fg_range
        self.bg_range = bg_range
        self.fg_nums = self.fg_range[1] - self.fg_range[0] + 1
        self.bg_nums = self.bg_range[1] - self.bg_range[0] + 1
        self.num_stages = num_stages
        self.inner_channels = inner_channels

        self.subnet = nn.ModuleList()
        for i in range(num_stages):
            self.subnet.append(
                PanFpnSubNet(
                    in_channels,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        self.score = nn.Conv2d(inner_channels, num_classes, 1)
        self.init_weights()

    def _set_things_to_void(self, gt_semantic_seg):
        gt_semantic_seg = gt_semantic_seg.int()
        fg_mask = (gt_semantic_seg >= self.fg_range[0]) * (
            gt_semantic_seg <= self.fg_range[1])
        bg_mask = (gt_semantic_seg >= self.bg_range[0]) * (
            gt_semantic_seg <= self.bg_range[1])

        new_gt_seg = fg_mask.int() * (self.bg_nums + 1)
        new_gt_seg = torch.where(bg_mask, gt_semantic_seg - self.fg_nums,
                                 new_gt_seg)
        return new_gt_seg

    def loss(self, logits, gt_semantic_seg):
        gt_semantic_seg = self._set_things_to_void(gt_semantic_seg)
        return super(PanopticFpnHead, self).loss(logits, gt_semantic_seg)

    def init_weights(self):
        nn.init.normal_(self.score.weight.data, 0, 0.01)
        self.score.bias.data.zero_()

    def forward(self, x):
        # the number of subnets must be equal with the length of features.
        assert self.num_stages == len(x)

        features = []
        for i, f in enumerate(x):
            f = self.subnet[i](f)
            features.append(f)

        features = torch.sum(torch.stack(features, dim=0), dim=0)
        score = self.score(features)
        ret = dict(fcn_score=score, fcn_feat=features)
        return ret
