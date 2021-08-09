import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from ..builder import HEADS
from ..utils import ConvUpsample
from .base_semantic_head import BaseSemanticHead


@HEADS.register_module()
class PanopticFPNHead(BaseSemanticHead):
    """PanopticFPNHead used in Panoptic FPN."""

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 inner_channels=128,
                 start_level=0,
                 end_level=4,
                 fg_range=(1, 80),
                 bg_range=(81, 133),
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 loss_semantic=dict(
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0)):
        super(PanopticFPNHead, self).__init__(num_classes, init_cfg,
                                              loss_semantic)
        self.fg_range = fg_range
        self.bg_range = bg_range
        self.fg_nums = self.fg_range[1] - self.fg_range[0] + 1
        self.bg_nums = self.bg_range[1] - self.bg_range[0] + 1
        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.num_stages = end_level - start_level
        self.inner_channels = inner_channels

        self.conv_upsample_layers = ModuleList()
        for i in range(start_level, end_level):
            self.conv_upsample_layers.append(
                ConvUpsample(
                    in_channels,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        self.conv_logits = nn.Conv2d(inner_channels, num_classes, 1)

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

    def loss(self, logits, gt_semantic_seg, label_bias=-1):
        # Merge thing classes to one class.
        gt_semantic_seg = self._set_things_to_void(gt_semantic_seg)
        return super().loss(logits, gt_semantic_seg, label_bias)

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.conv_logits.weight.data, 0, 0.01)
        self.conv_logits.bias.data.zero_()

    def forward(self, x):
        # the number of subnets must be not more than
        # the length of features.
        assert self.num_stages <= len(x)

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        logits = self.conv_logits(feats)
        ret = dict(logits=logits, feats=feats)
        return ret
