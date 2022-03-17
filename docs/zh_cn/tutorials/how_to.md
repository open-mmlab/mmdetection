# 教程 11: 常见用法集合
本教程主要是收集 MMDetection 中常见用法

## 使用 MMClassification 的骨干网络 

归功于 MMCV 中所提的层次注册器 (Hierarchy Registry) 功能，允许 OpenMMLab 的各个算法库中的模块相互调用。用户可以在 MMDetection 中使用 MMClassification 中的骨干网络，而无需为 MMDetection 也实现一个 MMClassification 中已经存在的网络。

假设想将 `MobileNetV3-small` 作为 `RetinaNet` 的骨干网络，则示例代码如下

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
        type='mmcls.MobileNetV3', # 使用 mmcls 中的 MobileNetV3
        arch='small',
        out_indices=(3, 8, 11), # 修改 out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')), # 为了确保预训练权重能正常加载，需要加 prefix
    # 修改 in_channels
    neck=dict(in_channels=[24, 48, 96], start_level=0))
```

由于 MMClassification 提供了 Py**T**orch **Im**age **M**odels (`timm`) 骨干网络的封装，用户可以通过 MMClassification 直接使用 `timm` 中的骨干网络。假设想将 [`EfficientNet-B1`](https://github.com/open-mmlab/mmdetection/blob/master/configs/timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco.py) 作为 `RetinaNet` 的骨干网络，则示例代码如下

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
        type='mmcls.TIMMBackbone', # 使用 mmcls 中 timm 骨干网络
        model_name='efficientnet_b1',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)), # 修改 out_indices
    neck=dict(in_channels=[24, 40, 112, 320])) # 修改 in_channels

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```
