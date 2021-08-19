import numpy as np
import torch
from mmcv import ConfigDict

from mmdet.models.dense_heads import CenterNet2Head


def test_centernet2_head_loss():
    """Tests centernet2 head loss when truth is empty and non-empty."""
    s = 256
    norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
    test_cfg = dict(topK=100, max_per_img=100)
    self = CenterNet2Head(
        num_classes=4,
        in_channels=1,
        feat_channels=4,
        test_cfg=test_cfg,
        norm_cfg=norm_cfg,
        dcn_on_last_conv=False)
    strides = [8, 16, 32, 64, 128]
    feat = [torch.rand(1, 1, s // stride, s // stride) for stride in strides]
    heatmap_out, reg_out = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    empty_gt_losses = self.loss(heatmap_out, reg_out, gt_bboxes)
    pos_loss = empty_gt_losses['pos_loss']
    neg_loss = empty_gt_losses['neg_loss']
    loss_bbox = empty_gt_losses['loss_bbox']
    assert pos_loss.item() == 0, 'pos_loss should be zero'
    assert neg_loss.item() > 0, 'neg_loss should be non-zero'
    assert loss_bbox.item() == 0, (
        'there should be no loss_bbox when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    one_gt_losses = self.loss(heatmap_out, reg_out, gt_bboxes)
    pos_loss = one_gt_losses['pos_loss']
    neg_loss = one_gt_losses['neg_loss']
    loss_bbox = one_gt_losses['loss_bbox']
    assert pos_loss.item() > 0, 'pos_loss should be non-zero'
    assert neg_loss.item() > 0, 'neg_loss should be non-zero'
    assert loss_bbox.item() > 0, 'loss_bbox should be non-zero'


def test_centernet2_head_get_bboxes():
    """Tests center head generating and decoding the heatmap."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': np.array([1., 1., 1., 1.]),
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s),
        'border': (0, 0, 0, 0),
        'flip': False
    }]

    gt_bboxes = [
        torch.Tensor([[10, 20, 200, 240], [40, 50, 100, 200],
                      [10, 20, 100, 240]])
    ]

    norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
    test_cfg = ConfigDict(
        nms_pre=100, max_per_img=100, nms=dict(type='nms', iou_threshold=0.9))
    self = CenterNet2Head(
        num_classes=4,
        in_channels=1,
        feat_channels=4,
        test_cfg=test_cfg,
        norm_cfg=norm_cfg,
        dcn_on_last_conv=False,
        original_dis_map=False)
    strides = [8, 16, 32, 64, 128]
    feats = [torch.rand(1, 1, s // stride, s // stride) for stride in strides]
    feature_map_sizes = [feat.size()[-2:] for feat in feats]
    points = self.get_points(feature_map_sizes, torch.long, 'cpu')
    pos_indices, reg_targets, flattened_hms =\
        self.get_targets(points, gt_bboxes)

    # Make sure center point targets match, ignoring neighbours
    flattened_strides = []
    for x in [[stride] * (s // stride)**2 for stride in strides]:
        flattened_strides += x
    reg_target_list = []
    gt_bbox_list = []
    for i in range(len(gt_bboxes[0])):
        bbox = gt_bboxes[0][i]
        stride = flattened_strides[pos_indices[i]]
        reg_target = reg_targets[pos_indices[i]] * stride
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        reg_target_list.append(
            (reg_target[0::2].sum(), reg_target[1::2].sum()))
        gt_bbox_list.append((w, h))
    reg_target_list.sort()
    gt_bbox_list.sort()
    assert pos_indices.size(0) == len(gt_bboxes[0])
    assert reg_target_list == gt_bbox_list

    # Test get_bboxes
    loc_per_level = [x[0] * x[1] for x in feature_map_sizes]
    flattened_hms = flattened_hms.split(loc_per_level)
    reg_targets = reg_targets.split(loc_per_level)
    hm_score = [
        x.reshape(1, 1, *i) for (x, i) in zip(flattened_hms, feature_map_sizes)
    ]
    reg_pred = [
        x.permute(1, 0).reshape(1, 4, *i)
        for (x, i) in zip(reg_targets, feature_map_sizes)
    ]
    detections = self.get_bboxes(
        hm_score, reg_pred, img_metas, cfg=test_cfg, rescale=True)
    out_bboxes = detections[0][:, 0:4]
    out_scores = detections[0][:, 4]
    out_bboxes = out_bboxes[:3, :].tolist()
    gt_bboxes = gt_bboxes[0].tolist()
    assert (out_scores[:3] == out_scores.max(-1)[0].expand(3)).all()
    assert out_bboxes.sort() == gt_bboxes.sort()
