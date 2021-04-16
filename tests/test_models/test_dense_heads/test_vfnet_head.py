import mmcv
import torch

from mmdet.models.dense_heads import VFNetHead


def test_vfnet_head_loss():
    """Tests vfnet head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    # since Focal Loss is not supported on CPU
    self = VFNetHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, loss_weight=1.0))
    if torch.cuda.is_available():
        self.cuda()
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
            for feat_size in [4, 8, 16, 32, 64]
        ]
        cls_scores, bbox_preds, bbox_preds_refine = self.forward(feat)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]
        gt_bboxes_ignore = None
        empty_gt_losses = self.loss(cls_scores, bbox_preds, bbox_preds_refine,
                                    gt_bboxes, gt_labels, img_metas,
                                    gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_loss.item() == 0, (
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = self.loss(cls_scores, bbox_preds, bbox_preds_refine,
                                  gt_bboxes, gt_labels, img_metas,
                                  gt_bboxes_ignore)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
