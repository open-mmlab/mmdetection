# Tutorial 11: Collection of common usage

This tutorial mainly collects common usages in MMDetection

## Backbone network using MMClassification

The Hierarchy Registry function proposed in MMCV allows modules in various algorithm libraries of OpenMMLab to call each other. Users can use backbone networks from MMClassification in MMDetection without implementing a network that already exists in MMClassification for MMDetection as well.

Suppose you want to use `MobileNetV3-small` as the backbone network of `RetinaNet`, the sample code is as follows.

```python
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# please install mmcls>=0.20.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
pretrained = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.MobileNetV3', # Using MobileNetV3 from mmcls
        arch='small',
        out_indices=(3, 8, 11), # Modify out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')), # Need to add prefix to ensure that pre-training weights can be loaded normally
    # Modify in_channels
    neck=dict(in_channels=[24, 48, 96], start_level=0))
```

Since MMClassification provides a wrapper for the Py**T**orch **Im**age **M**odels (`timm`) backbone network, users can directly use the backbone network in `timm` through MMClassification. Suppose you want to use [`EfficientNet-B1`](https://github.com/open-mmlab/mmdetection/blob/master/configs/timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco.py) as the backbone network of `RetinaNet`, the sample code is as follows.

```python
# https://github.com/open-mmlab/mmdetection/blob/master/configs/timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco.py

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# please install mmcls>=0.20.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.TIMMBackbone', # Using timm from mmcls
        model_name='efficientnet_b1',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)), # Modify out_indices
    neck=dict(in_channels=[24, 40, 112, 320])) # Modify in_channels

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```
