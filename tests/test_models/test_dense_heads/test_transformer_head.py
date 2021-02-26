import torch

from mmdet.models.dense_heads import TransformerHead


def test_transformer_head_loss():
    """Tests transformer head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s)
    }]
    train_cfg = dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)))
    transformer_cfg = dict(
        type='Transformer',
        embed_dims=4,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_channels=1,
        dropout=0.1,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='LN'),
        num_fcs=2,
        pre_norm=False,
        return_intermediate_dec=True)
    positional_encoding_cfg = dict(
        type='SinePositionalEncoding', num_feats=2, normalize=True)
    self = TransformerHead(
        num_classes=4,
        in_channels=1,
        num_fcs=2,
        train_cfg=train_cfg,
        transformer=transformer_cfg,
        positional_encoding=positional_encoding_cfg)
    self.init_weights()
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds = self.forward(feat, img_metas)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    for key, loss in empty_gt_losses.items():
        if 'cls' in key:
            assert loss.item() > 0, 'cls loss should be non-zero'
        elif 'bbox' in key:
            assert loss.item(
            ) == 0, 'there should be no box loss when there are no true boxes'
        elif 'iou' in key:
            assert loss.item(
            ) == 0, 'there should be no iou loss when there are no true boxes'

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    for loss in one_gt_losses.values():
        assert loss.item(
        ) > 0, 'cls loss, or box loss, or iou loss should be non-zero'

    # test forward_train
    self.forward_train(feat, img_metas, gt_bboxes, gt_labels)

    # test inference mode
    self.get_bboxes(cls_scores, bbox_preds, img_metas, rescale=True)
