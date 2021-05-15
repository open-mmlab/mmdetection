import mmcv
import torch

from mmdet.models.dense_heads import FCOSHead


def test_fcos_head_loss():
    """Tests fcos head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    # since Focal Loss is not supported on CPU
    self = FCOSHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds, centerness = self.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, centerness, gt_bboxes,
                                gt_labels, img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, centerness, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
