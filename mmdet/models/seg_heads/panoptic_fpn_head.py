# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import ConvUpsample
from ..utils import interpolate_as
from .base_semantic_head import BaseSemanticHead


@MODELS.register_module()
class PanopticFPNHead(BaseSemanticHead):
    """PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.

    Arg:
        num_things_classes (int): Number of thing classes. Default: 80.
        num_stuff_classes (int): Number of stuff classes. Default: 53.
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        conv_cfg (Optional[Union[ConfigDict, dict]]): Dictionary to construct
            and config conv layer.
        norm_cfg (Union[ConfigDict, dict]): Dictionary to construct and config
            norm layer. Use ``GN`` by default.
        init_cfg (Optional[Union[ConfigDict, dict]]): Initialization config
            dict.
        loss_seg (Union[ConfigDict, dict]): the loss of the semantic head.
    """

    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 in_channels: int = 256,
                 inner_channels: int = 128,
                 start_level: int = 0,
                 end_level: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_seg: ConfigType = dict(
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        seg_rescale_factor = 1 / 2**(start_level + 2)
        super().__init__(
            num_classes=num_stuff_classes + 1,
            seg_rescale_factor=seg_rescale_factor,
            loss_seg=loss_seg,
            init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
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

    def _set_things_to_void(self, gt_semantic_seg: Tensor) -> Tensor:
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

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Dict[str, Tensor]: The loss of semantic head.
        """
        seg_preds = self(x)['seg_preds']
        gt_semantic_segs = [
            data_sample.gt_sem_seg.sem_seg
            for data_sample in batch_data_samples
        ]

        gt_semantic_segs = torch.stack(gt_semantic_segs)
        if self.seg_rescale_factor != 1.0:
            gt_semantic_segs = F.interpolate(
                gt_semantic_segs.float(),
                scale_factor=self.seg_rescale_factor,
                mode='nearest').squeeze(1)

        # Things classes will be merged to one class in PanopticFPN.
        gt_semantic_segs = self._set_things_to_void(gt_semantic_segs)

        if seg_preds.shape[-2:] != gt_semantic_segs.shape[-2:]:
            seg_preds = interpolate_as(seg_preds, gt_semantic_segs)
        seg_preds = seg_preds.permute((0, 2, 3, 1))

        loss_seg = self.loss_seg(
            seg_preds.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_segs.reshape(-1).long())

        return dict(loss_seg=loss_seg)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        nn.init.normal_(self.conv_logits.weight.data, 0, 0.01)
        self.conv_logits.bias.data.zero_()

    def forward(self, x: Tuple[Tensor]) -> Dict[str, Tensor]:
        """Forward.

        Args:
            x (Tuple[Tensor]): Multi scale Feature maps.

        Returns:
            dict[str, Tensor]: semantic segmentation predictions and
                feature maps.
        """
        # the number of subnets must be not more than
        # the length of features.
        assert self.num_stages <= len(x)

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        seg_feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        seg_preds = self.conv_logits(seg_feats)
        out = dict(seg_preds=seg_preds, seg_feats=seg_feats)
        return out
