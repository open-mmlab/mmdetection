import unittest

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmdet.models.dense_heads import RepPointsHead
from mmdet.structures import DetDataSample


class TestRepPointsHead(unittest.TestCase):

    @parameterized.expand(['moment', 'minmax', 'partial_minmax'])
    def test_head_loss(self, transform_method='moment'):
        cfg = ConfigDict(
            dict(
                num_classes=2,
                in_channels=32,
                point_feat_channels=10,
                num_points=9,
                gradient_mul=0.1,
                point_strides=[8, 16, 32, 64, 128],
                point_base_scale=4,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox_init=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                loss_bbox_refine=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                use_grid_points=False,
                center_init=True,
                transform_method=transform_method,
                moment_mul=0.01,
                init_cfg=dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='reppoints_cls_out',
                        std=0.01,
                        bias_prob=0.01)),
                train_cfg=dict(
                    init=dict(
                        assigner=dict(
                            type='PointAssigner', scale=4, pos_num=1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    refine=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                            ignore_iof_thr=-1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False)),
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)))
        reppoints_head = RepPointsHead(**cfg)
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }]
        x = [
            torch.rand(1, 32, s // 2**(i + 2), s // 2**(i + 2))
            for i in range(5)
        ]

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_bboxes_ignore = None

        reppoints_head.train()
        forward_outputs = reppoints_head.forward(x)
        empty_gt_losses = reppoints_head.loss_by_feat(*forward_outputs,
                                                      [gt_instances],
                                                      img_metas,
                                                      gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no pts loss.
        for key, losses in empty_gt_losses.items():
            for loss in losses:
                if 'cls' in key:
                    self.assertGreater(loss.item(), 0,
                                       'cls loss should be non-zero')
                elif 'pts' in key:
                    self.assertEqual(
                        loss.item(), 0,
                        'there should be no reg loss when no ground true boxes'
                    )

        # When truth is non-empty then both cls and pts loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        one_gt_losses = reppoints_head.loss_by_feat(*forward_outputs,
                                                    [gt_instances], img_metas,
                                                    gt_bboxes_ignore)
        # loss_cls should all be non-zero
        self.assertTrue(
            all([loss.item() > 0 for loss in one_gt_losses['loss_cls']]))
        # only one level loss_pts_init is non-zero
        cnt_non_zero = 0
        for loss in one_gt_losses['loss_pts_init']:
            if loss.item() != 0:
                cnt_non_zero += 1
        self.assertEqual(cnt_non_zero, 1)

        # only one level loss_pts_refine is non-zero
        cnt_non_zero = 0
        for loss in one_gt_losses['loss_pts_init']:
            if loss.item() != 0:
                cnt_non_zero += 1
        self.assertEqual(cnt_non_zero, 1)

        # test loss
        samples = DetDataSample()
        samples.set_metainfo(img_metas[0])
        samples.gt_instances = gt_instances
        reppoints_head.loss(x, [samples])
        # test only predict
        reppoints_head.eval()
        reppoints_head.predict(x, [samples], rescale=True)
