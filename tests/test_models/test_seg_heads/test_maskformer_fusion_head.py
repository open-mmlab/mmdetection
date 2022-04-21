import pytest
import torch
from mmcv import ConfigDict

from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead


def test_maskformer_fusion_head():
    img_metas = [
        {
            'batch_input_shape': (128, 160),
            'img_shape': (126, 160, 3),
            'ori_shape': (63, 80, 3),
            'pad_shape': (128, 160, 3)
        },
    ]
    num_things_classes = 80
    num_stuff_classes = 53
    num_classes = num_things_classes + num_stuff_classes
    config = ConfigDict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        test_cfg=dict(
            panoptic_on=True,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            object_mask_thr=0.8,
            iou_thr=0.8,
            filter_low_score=False),
        init_cfg=None)

    self = MaskFormerFusionHead(**config)

    # test forward_train
    assert self.forward_train() == dict()

    mask_cls_results = torch.rand((1, 100, num_classes + 1))
    mask_pred_results = torch.rand((1, 100, 128, 160))

    # test panoptic_postprocess and instance_postprocess
    results = self.simple_test(mask_cls_results, mask_pred_results, img_metas)
    assert 'ins_results' in results[0] and 'pan_results' in results[0]

    # test semantic_postprocess
    config.test_cfg.semantic_on = True
    with pytest.raises(AssertionError):
        self.simple_test(mask_cls_results, mask_pred_results, img_metas)

    with pytest.raises(NotImplementedError):
        self.semantic_postprocess(mask_cls_results, mask_pred_results)
