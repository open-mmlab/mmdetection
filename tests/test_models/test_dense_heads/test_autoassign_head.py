import mmcv
import torch

from mmdet.models.dense_heads.autoassign_head import AutoAssignHead
from mmdet.models.dense_heads.paa_head import levels_to_images


def test_autoassign_head_loss():
    """Tests autoassign head loss when truth is empty and non-empty."""

    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(assigner=None, allowed_border=-1, pos_weight=-1, debug=False))
    self = AutoAssignHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    self.init_weights()
    cls_scores, bbox_preds, objectnesses = self(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses,
                                gt_bboxes, gt_labels, img_metas,
                                gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_pos_loss = empty_gt_losses['loss_pos']
    empty_neg_loss = empty_gt_losses['loss_neg']
    empty_center_loss = empty_gt_losses['loss_center']
    assert empty_neg_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_pos_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_center_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_pos_loss = one_gt_losses['loss_pos']
    onegt_neg_loss = one_gt_losses['loss_neg']
    onegt_center_loss = one_gt_losses['loss_center']
    assert onegt_pos_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_neg_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_center_loss.item() > 0, 'box loss should be non-zero'
    n, c, h, w = 10, 4, 20, 20
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    cls_scores = [torch.ones(2, 4, 5, 5)]
    bbox_preds = [torch.ones(2, 4, 5, 5)]
    iou_preds = [torch.ones(2, 1, 5, 5)]
    mlvl_anchors = [torch.ones(5 * 5, 4)]
    img_shape = None
    scale_factor = [0.5, 0.5]
    cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))
    rescale = False
    self._get_bboxes(
        cls_scores,
        bbox_preds,
        iou_preds,
        mlvl_anchors,
        img_shape,
        scale_factor,
        cfg,
        rescale=rescale)
