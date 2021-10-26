# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.roi_heads.mask_heads import (DynamicMaskHead, FCNMaskHead,
                                               MaskIoUHead)
from .utils import _dummy_bbox_sampling


def test_mask_head_loss():
    """Test mask head loss when mask target is empty."""
    self = FCNMaskHead(
        num_convs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        num_classes=8)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    # create dummy mask
    import numpy as np
    from mmdet.core import BitmapMasks
    dummy_mask = np.random.randint(0, 2, (1, 160, 240), dtype=np.uint8)
    gt_masks = [BitmapMasks(dummy_mask, 160, 240)]

    # create dummy train_cfg
    train_cfg = mmcv.Config(dict(mask_size=12, mask_thr_binary=0.5))

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8, 6, 6)

    mask_pred = self.forward(dummy_feats)
    mask_targets = self.get_targets(sampling_results, gt_masks, train_cfg)
    pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
    loss_mask = self.loss(mask_pred, mask_targets, pos_labels)

    onegt_mask_loss = sum(loss_mask['loss_mask'])
    assert onegt_mask_loss.item() > 0, 'mask loss should be non-zero'

    # test mask_iou_head
    mask_iou_head = MaskIoUHead(
        num_convs=1,
        num_fcs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        fc_out_channels=8,
        num_classes=8)

    pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
    mask_iou_pred = mask_iou_head(dummy_feats, pos_mask_pred)
    pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

    mask_iou_targets = mask_iou_head.get_targets(sampling_results, gt_masks,
                                                 pos_mask_pred, mask_targets,
                                                 train_cfg)
    loss_mask_iou = mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
    onegt_mask_iou_loss = loss_mask_iou['loss_mask_iou'].sum()
    assert onegt_mask_iou_loss.item() >= 0

    # test dynamic_mask_head
    dummy_proposal_feats = torch.rand(num_sampled, 8)
    dynamic_mask_head = DynamicMaskHead(
        dynamic_conv_cfg=dict(
            type='DynamicConv',
            in_channels=8,
            feat_channels=8,
            out_channels=8,
            input_feat_shape=6,
            with_proj=False,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN')),
        num_convs=1,
        num_classes=8,
        in_channels=8,
        roi_feat_size=6)

    mask_pred = dynamic_mask_head(dummy_feats, dummy_proposal_feats)

    mask_target = dynamic_mask_head.get_targets(sampling_results, gt_masks,
                                                train_cfg)
    loss_mask = dynamic_mask_head.loss(mask_pred, mask_target, pos_labels)
    loss_mask = loss_mask['loss_mask'].sum()
    assert loss_mask.item() >= 0
