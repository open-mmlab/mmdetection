import mmcv
import torch

from mmdet.models.dense_heads import SOLOHead


def test_solo_head_loss():
    """Tests solo head loss when truth is empty and non-empty."""
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
    self = SOLOHead(
        num_classes=4,
        in_channels=1,
        num_grids=[40, 36, 24, 16, 12],
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        train_cfg=train_cfg)
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    mask_preds, cls_preds = self.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_masks = [torch.empty((0, 550, 550))]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(
        mask_preds,
        cls_preds,
        gt_labels,
        gt_masks,
        img_metas,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_mask_loss = empty_gt_losses['loss_mask']
    empty_cls_loss = empty_gt_losses['loss_cls']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_mask_loss.item() == 0, (
        'there should be no mask loss when there are no true masks')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    gt_masks = [(torch.rand((1, 256, 256)) > 0.5).float()]
    one_gt_losses = self.loss(
        mask_preds,
        cls_preds,
        gt_labels,
        gt_masks,
        img_metas,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore)
    onegt_mask_loss = one_gt_losses['loss_mask']
    onegt_cls_loss = one_gt_losses['loss_cls']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_mask_loss.item() > 0, 'mask loss should be non-zero'
