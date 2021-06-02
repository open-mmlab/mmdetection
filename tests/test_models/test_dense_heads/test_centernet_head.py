import numpy as np
import torch
from mmcv import ConfigDict

from mmdet.models.dense_heads import CenterNetHead


def test_center_head_loss():
    """Tests center head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    test_cfg = dict(topK=100, max_per_img=100)
    self = CenterNetHead(
        num_classes=4, in_channel=1, feat_channel=4, test_cfg=test_cfg)

    feat = [torch.rand(1, 1, s, s)]
    center_out, wh_out, offset_out = self.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(center_out, wh_out, offset_out, gt_bboxes,
                                gt_labels, img_metas, gt_bboxes_ignore)
    loss_center = empty_gt_losses['loss_center_heatmap']
    loss_wh = empty_gt_losses['loss_wh']
    loss_offset = empty_gt_losses['loss_offset']
    assert loss_center.item() > 0, 'loss_center should be non-zero'
    assert loss_wh.item() == 0, (
        'there should be no loss_wh when there are no true boxes')
    assert loss_offset.item() == 0, (
        'there should be no loss_offset when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(center_out, wh_out, offset_out, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    loss_center = one_gt_losses['loss_center_heatmap']
    loss_wh = one_gt_losses['loss_wh']
    loss_offset = one_gt_losses['loss_offset']
    assert loss_center.item() > 0, 'loss_center should be non-zero'
    assert loss_wh.item() > 0, 'loss_wh should be non-zero'
    assert loss_offset.item() > 0, 'loss_offset should be non-zero'


def test_centernet_head_get_bboxes():
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
    test_cfg = ConfigDict(
        dict(topk=100, local_maximum_kernel=3, max_per_img=100))
    gt_bboxes = [
        torch.Tensor([[10, 20, 200, 240], [40, 50, 100, 200],
                      [10, 20, 100, 240]])
    ]
    gt_labels = [torch.LongTensor([1, 1, 2])]

    self = CenterNetHead(
        num_classes=4, in_channel=1, feat_channel=4, test_cfg=test_cfg)
    self.feat_shape = (1, 1, s // 4, s // 4)
    targets, _ = self.get_targets(gt_bboxes, gt_labels, self.feat_shape,
                                  img_metas[0]['pad_shape'])
    center_target = targets['center_heatmap_target']
    wh_target = targets['wh_target']
    offset_target = targets['offset_target']
    # make sure assign target right
    for i in range(len(gt_bboxes[0])):
        bbox, label = gt_bboxes[0][i] / 4, gt_labels[0][i]
        ctx, cty = sum(bbox[0::2]) / 2, sum(bbox[1::2]) / 2
        int_ctx, int_cty = int(sum(bbox[0::2]) / 2), int(sum(bbox[1::2]) / 2)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x_off = ctx - int(ctx)
        y_off = cty - int(cty)
        assert center_target[0, label, int_cty, int_ctx] == 1
        assert wh_target[0, 0, int_cty, int_ctx] == w
        assert wh_target[0, 1, int_cty, int_ctx] == h
        assert offset_target[0, 0, int_cty, int_ctx] == x_off
        assert offset_target[0, 1, int_cty, int_ctx] == y_off
    # make sure get_bboxes is right
    detections = self.get_bboxes([center_target], [wh_target], [offset_target],
                                 img_metas,
                                 rescale=True,
                                 with_nms=False)
    out_bboxes = detections[0][0][:3]
    out_clses = detections[0][1][:3]
    for bbox, cls in zip(out_bboxes, out_clses):
        flag = False
        for gt_bbox, gt_cls in zip(gt_bboxes[0], gt_labels[0]):
            if (bbox[:4] == gt_bbox[:4]).all():
                flag = True
        assert flag, 'get_bboxes is wrong'
