# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32

from ..builder import build_loss
from ..utils import interpolate_as


class BaseSemanticHead(BaseModule, metaclass=ABCMeta):
    """Base module of Semantic Head.

    Args:
        num_classes (int): the number of classes.
        init_cfg (dict): the initialization config.
        loss_seg (dict): the loss of the semantic head.
    """

    def __init__(self,
                 num_classes,
                 init_cfg=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0)):
        super(BaseSemanticHead, self).__init__(init_cfg)
        self.loss_seg = build_loss(loss_seg)
        self.num_classes = num_classes

    @force_fp32(apply_to=('seg_preds', ))
    def loss(self, seg_preds, gt_semantic_seg):
        """Get the loss of semantic head.

        Args:
            seg_preds (Tensor): The input logits with the shape (N, C, H, W).
            gt_semantic_seg: The ground truth of semantic segmentation with
                the shape (N, H, W).
            label_bias: The starting number of the semantic label.
                Default: 1.

        Returns:
            dict: the loss of semantic head.
        """
        if seg_preds.shape[-2:] != gt_semantic_seg.shape[-2:]:
            seg_preds = interpolate_as(seg_preds, gt_semantic_seg)
        seg_preds = seg_preds.permute((0, 2, 3, 1))

        loss_seg = self.loss_seg(
            seg_preds.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_seg.reshape(-1).long())
        return dict(loss_seg=loss_seg)

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """
        pass

    def forward_train(self, x, gt_semantic_seg):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        return self.loss(seg_preds, gt_semantic_seg)

    def simple_test(self, x, img_metas, rescale=False):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        seg_preds = F.interpolate(
            seg_preds,
            size=img_metas[0]['pad_shape'][:2],
            mode='bilinear',
            align_corners=False)

        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            seg_preds = seg_preds[:, :, :h, :w]

            h, w, _ = img_metas[0]['ori_shape']
            seg_preds = F.interpolate(
                seg_preds, size=(h, w), mode='bilinear', align_corners=False)
        return seg_preds
