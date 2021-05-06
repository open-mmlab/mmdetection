import mmcv
import torch

from mmdet.models.dense_heads import SABLRetinaHead


def test_sabl_retina_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    head = SABLRetinaHead(
        num_classes=4,
        in_channels=3,
        feat_channels=10,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        train_cfg=cfg)
    if torch.cuda.is_available():
        head.cuda()
        # Anchor head expects a multiple levels of features per image
        feat = [
            torch.rand(1, 3, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
            for i in range(len(head.approx_anchor_generator.base_anchors))
        ]
        cls_scores, bbox_preds = head.forward(feat)

        # Test that empty ground truth encourages the network
        # to predict background
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]

        gt_bboxes_ignore = None
        empty_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes,
                                    gt_labels, img_metas, gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_cls_loss = sum(empty_gt_losses['loss_bbox_cls'])
        empty_box_reg_loss = sum(empty_gt_losses['loss_bbox_reg'])
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_cls_loss.item() == 0, (
            'there should be no box cls loss when there are no true boxes')
        assert empty_box_reg_loss.item() == 0, (
            'there should be no box reg loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should
        # be nonzero for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                  img_metas, gt_bboxes_ignore)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_cls_loss = sum(one_gt_losses['loss_bbox_cls'])
        onegt_box_reg_loss = sum(one_gt_losses['loss_bbox_reg'])
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_cls_loss.item() > 0, 'box loss cls should be non-zero'
        assert onegt_box_reg_loss.item() > 0, 'box loss reg should be non-zero'
