import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss, build_neck


@HEADS.register_module()
class SemanticHead(nn.Module):
    """Semantic segmentation head that can be used in panoptic segmentation.

    Args:
        semantic_decoder (dict): Config dict of decoder.
            It usually is a neck, like semantic FPN.
        in_channels (int, optional): Input channels. Defaults to 256.
        num_classes (int, optional):  Number of semantic classes including
            the background. Defaults to 183.
        ignore_label (int, optional): Labels to be ignored. Defaults to 255.
        loss_seg (dict, optional): Config dict of loss.
            Defaults to `dict(type='CrossEntropyLoss', use_sigmoid=False, \
            loss_weight=1.0)`.
        conv_cfg (dict, optional): Config of convolutional layers.
            Defaults to None.
        norm_cfg (dict, optional): Config of normalization layers.
            Defaults to None.
    """

    def __init__(self,
                 semantic_decoder,
                 in_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticHead, self).__init__()
        self.semantic_decoder = build_neck(semantic_decoder)
        self.conv_logits = nn.Conv2d(in_channels, num_classes, 1)
        self.loss_seg = build_loss(loss_seg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        x = self.semantic_decoder(feats)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.loss_seg(
            mask_pred, labels, ignore_index=self.ignore_label)
        return loss_semantic_seg

    def get_semantic_seg(self, seg_preds, ori_shape, img_shape_withoutpad):
        """Obtain semantic segmentation map for panoptic segmentation.

        Args:
            seg_preds (torch.Tensor): Segmentation prediction
            ori_shape (tuple[int]): Input image shape with padding.
            img_shape_withoutpad (tuple[int]): Original image shape before
                without padding.

        Returns:
            list[list[np.ndarray]]: The decoded segmentation masks.
                The first dimension is the number of classes.
                The second dimension is the number of masks of a similar class.
        """
        # only surport 1 batch
        seg_preds = seg_preds[:, :, 0:img_shape_withoutpad[0],
                              0:img_shape_withoutpad[1]]
        seg_masks = F.softmax(seg_preds, 1)
        seg_masks = F.interpolate(
            seg_masks,
            size=ori_shape[0:2],
            mode='bilinear',
            align_corners=False)
        seg_masks = torch.max(seg_masks, 1).indices
        seg_masks = seg_masks.float()
        seg_masks = seg_masks[0]

        seg_masks = seg_masks.cpu().numpy()
        unique_seg_masks = np.unique(seg_masks).astype(np.int)
        seg_results = [[] for _ in range(self.num_classes - 1)]

        for i in unique_seg_masks:
            if i == 0:
                continue
            cls_mask = np.zeros((ori_shape[0], ori_shape[1])).astype(np.uint8)
            cls_mask[seg_masks == i] = 1
            rle = mask_util.encode(
                np.array(cls_mask[:, :, np.newaxis], order='F'))[0]
            seg_results[i - 1].append(rle)

        return seg_results
