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
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0)):
        super(BaseSemanticHead, self).__init__(init_cfg)
        self.loss_seg = build_loss(loss_seg)
        self.num_classes = num_classes

    @force_fp32(apply_to=('logits', ))
    def loss(self, logits, gt_semantic_seg, label_bias=1):
        """Get the loss of semantic head.

        Args:
            logits (Tensor): The input logits with the shape (N, C, H, W).
            gt_semantic_seg: The ground truth of semantic segmentation with
                the shape (N, H, W).
            label_bias: The starting number of the semantic label.
                Default: 1.

        Returns:
            dict: the loss of semantic head.
        """
        if logits.shape[-2:] != gt_semantic_seg.shape[-2:]:
            logits = interpolate_as(logits, gt_semantic_seg)
        logits = logits.permute((0, 2, 3, 1))
        # make the semantic label start from 0
        gt_semantic_seg = gt_semantic_seg - label_bias

        loss_seg = self.loss_seg(
            logits.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_seg.reshape(-1).long())
        return dict(loss_seg=loss_seg)

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'logits'
                and 'feats'.
        """
        pass

    def forward_train(self, x, gt_semantic_seg):
        output = self.forward(x)
        logits = output['logits']
        return self.loss(logits, gt_semantic_seg)

    def simple_test(self, x, img_metas, rescale=False):
        output = self.forward(x)
        logits = output['logits']
        logits = F.interpolate(
            logits,
            size=img_metas[0]['pad_shape'][:2],
            mode='bilinear',
            align_corners=False)

        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            logits = logits[:, :, :h, :w]

            h, w, _ = img_metas[0]['ori_shape']
            logits = F.interpolate(
                logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits
