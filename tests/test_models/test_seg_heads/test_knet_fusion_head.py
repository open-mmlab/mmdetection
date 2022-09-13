import pytest
import torch
from mmcv import ConfigDict

from mmdet.models.seg_heads.panoptic_fusion_heads import KNetFusionHead


def test_knet_fusion_head():
    img_metas = [
        {
            'batch_input_shape': (128, 160),
            'img_shape': (126, 160, 3),
            'ori_shape': (63, 80, 3),
            'pad_shape': (128, 160, 3)
        },
    ]
    num_things_classes = 8
    num_stuff_classes = 5
    num_proposals = 10
    num_classes = num_things_classes + num_stuff_classes
    config = ConfigDict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_proposals=num_proposals,
        loss_panoptic=None,
        test_cfg=dict(
            # instance segmentation and panoptic segmentation
            # can't be turn on at the same time
            panoptic_on=True,
            instance_on=True,
            # test cfg for panoptic segmentation
            overlap_thr=0.6,
            instance_score_thr=0.3,
            # tes cfg for instance segmentation
            max_per_img=num_proposals,
            mask_thr=0.5),
        init_cfg=None)

    self = KNetFusionHead(**config)

    # test forward_train
    assert self.forward_train() == dict()

    mask_cls_results = torch.rand((1, num_proposals, num_classes + 1))
    mask_pred_results = torch.rand((1, num_proposals, 128, 160))

    # instance segmentation and panoptic segmentation
    # can't be turn on at the same time
    with pytest.raises(AssertionError):
        self.simple_test(mask_cls_results, mask_pred_results, img_metas)

    self.test_cfg.panoptic_on = False
    self.test_cfg.instance_on = False
    with pytest.raises(AssertionError):
        self.simple_test(mask_cls_results, mask_pred_results, img_metas)

    # test panoptic_postprocess
    self.test_cfg.panoptic_on = True

    results = self.simple_test(mask_cls_results, mask_pred_results, img_metas)
    assert 'pan_results' in results[0]

    # test instance_postprocess
    self.test_cfg.panoptic_on = False
    self.test_cfg.instance_on = True
    mask_cls_results = torch.rand((1, num_proposals, num_things_classes))
    results = self.simple_test(mask_cls_results, mask_pred_results, img_metas)
    assert 'ins_results' in results[0]
