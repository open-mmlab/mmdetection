# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from ..builder import HEADS
from ..utils import ConvUpsample
from .base_semantic_head import BaseSemanticHead


@HEADS.register_module()
class PanopticFPNHead(BaseSemanticHead):
    """PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.

    Arg:
        num_things_classes (int): Number of thing classes. Default: 80.
        num_stuff_classes (int): Number of stuff classes. Default: 53.
        num_classes (int): Number of classes, including all stuff
            classes and one thing class. This argument is deprecated,
            please use ``num_things_classes`` and ``num_stuff_classes``.
            The module will automatically infer the num_classes by
            ``num_stuff_classes + 1``.
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        fg_range (tuple): Range of the foreground classes. It starts
            from ``0`` to ``num_things_classes-1``. Deprecated, please use
             ``num_things_classes`` directly.
        bg_range (tuple): Range of the background classes. It starts
            from ``num_things_classes`` to ``num_things_classes +
            num_stuff_classes - 1``. Deprecated, please use
            ``num_stuff_classes`` and ``num_things_classes`` directly.
        conv_cfg (dict): Dictionary to construct and config
            conv layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Use ``GN`` by default.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        loss_seg (dict): the loss of the semantic head.
    """

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_classes=None,
                 in_channels=256,
                 inner_channels=128,
                 start_level=0,
                 end_level=4,
                 fg_range=None,
                 bg_range=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0)):
        if num_classes is not None:
            warnings.warn(
                '`num_classes` is deprecated now, please set '
                '`num_stuff_classes` directly, the `num_classes` will be '
                'set to `num_stuff_classes + 1`')
            # num_classes = num_stuff_classes + 1 for PanopticFPN.
            assert num_classes == num_stuff_classes + 1
        super(PanopticFPNHead, self).__init__(num_stuff_classes + 1, init_cfg,
                                              loss_seg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        if fg_range is not None and bg_range is not None:
            self.fg_range = fg_range
            self.bg_range = bg_range
            self.num_things_classes = fg_range[1] - fg_range[0] + 1
            self.num_stuff_classes = bg_range[1] - bg_range[0] + 1
            warnings.warn(
                '`fg_range` and `bg_range` are deprecated now, '
                f'please use `num_things_classes`={self.num_things_classes} '
                f'and `num_stuff_classes`={self.num_stuff_classes} instead.')

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
        self.conv_logits = nn.Conv2d(inner_channels, self.num_classes, 1)

    def _set_things_to_void(self, gt_semantic_seg):
        """Merge thing classes to one class.

        In PanopticFPN, the background labels will be reset from `0` to
        `self.num_stuff_classes-1`, the foreground labels will be merged to
        `self.num_stuff_classes`-th channel.
        """
        gt_semantic_seg = gt_semantic_seg.int()
        fg_mask = gt_semantic_seg < self.num_things_classes
        bg_mask = (gt_semantic_seg >= self.num_things_classes) * (
            gt_semantic_seg < self.num_things_classes + self.num_stuff_classes)

        new_gt_seg = torch.clone(gt_semantic_seg)
        new_gt_seg = torch.where(bg_mask,
                                 gt_semantic_seg - self.num_things_classes,
                                 new_gt_seg)
        new_gt_seg = torch.where(fg_mask,
                                 fg_mask.int() * self.num_stuff_classes,
                                 new_gt_seg)
        return new_gt_seg

    def loss(self, seg_preds, gt_semantic_seg):
        """The loss of PanopticFPN head.

        Things classes will be merged to one class in PanopticFPN.
        """
        gt_semantic_seg = self._set_things_to_void(gt_semantic_seg)
        return super().loss(seg_preds, gt_semantic_seg)

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
        seg_preds = self.conv_logits(feats)
        out = dict(seg_preds=seg_preds, feats=feats)
        return out
