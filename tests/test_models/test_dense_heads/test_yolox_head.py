import mmcv
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmdet.models.dense_heads import YOLOXHead


def test_yolox_head_loss():
    """Tests yolox head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='SimOTAAssigner',
                center_radius=2.5,
                candidate_topk=10,
                iou_weight=3.0,
                cls_weight=1.0)))
    self = YOLOXHead(
        num_classes=4, in_channels=1, use_depthwise=False, train_cfg=train_cfg)
    assert not self.use_l1
    assert isinstance(self.multi_level_cls_convs[0][0], ConvModule)

    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16]
    ]
    cls_scores, bbox_preds, objectnesses = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    empty_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses,
                                gt_bboxes, gt_labels, img_metas)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls'].sum()
    empty_box_loss = empty_gt_losses['loss_bbox'].sum()
    empty_obj_loss = empty_gt_losses['loss_obj'].sum()
    assert empty_cls_loss.item() == 0, (
        'there should be no cls loss when there are no true boxes')
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_obj_loss.item() > 0, 'objectness loss should be non-zero'

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    self = YOLOXHead(
        num_classes=4, in_channels=1, use_depthwise=True, train_cfg=train_cfg)
    assert isinstance(self.multi_level_cls_convs[0][0],
                      DepthwiseSeparableConvModule)
    self.use_l1 = True
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses, gt_bboxes,
                              gt_labels, img_metas)
    onegt_cls_loss = one_gt_losses['loss_cls'].sum()
    onegt_box_loss = one_gt_losses['loss_bbox'].sum()
    onegt_obj_loss = one_gt_losses['loss_obj'].sum()
    onegt_l1_loss = one_gt_losses['loss_l1'].sum()
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_obj_loss.item() > 0, 'obj loss should be non-zero'
    assert onegt_l1_loss.item() > 0, 'l1 loss should be non-zero'
