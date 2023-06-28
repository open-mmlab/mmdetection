This tutorial collects answers to any `How to xxx with MMDetection`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

# Use backbone network through MMPretrain

The model registry in MMDet, MMPreTrain, MMSeg all inherit from the root registry in MMEngine. This allows these repositories to directly use the modules already implemented by each other. Therefore, users can use backbone networks from MMPretrain in MMDetection without implementing a network that already exists in MMPretrain.

## Use backbone network implemented in MMPretrain

Suppose you want to use `MobileNetV3-small` as the backbone network of `RetinaNet`, the example config is as the following.

```python
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
pretrained = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'
model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmpretrain.MobileNetV3', # Using MobileNetV3 from mmpretrain
        arch='small',
        out_indices=(3, 8, 11), # Modify out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')), # The pre-trained weights of backbone network in mmpretrain have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
    # Modify in_channels
    neck=dict(in_channels=[24, 48, 96], start_level=0))
```

## Use backbone network in TIMM through MMPretrain

MMPretrain also provides a wrapper for the PyTorch Image Models (timm) backbone network, users can directly use the backbone network in timm through MMPretrain. Suppose you want to use [EfficientNet-B1](../../../configs/timm_example/retinanet_timm-efficientnet-b1_fpn_1x_coco.py) as the backbone network of RetinaNet, the example config is as the following.

```python
# https://github.com/open-mmlab/mmdetection/blob/main/configs/timm_example/retinanet_timm-efficientnet-b1_fpn_1x_coco.py

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmpretrain.TIMMBackbone', # Using timm from mmpretrain
        model_name='efficientnet_b1',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)), # Modify out_indices
    neck=dict(in_channels=[24, 40, 112, 320])) # Modify in_channels

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

`type='mmpretrain.TIMMBackbone'` means use the `TIMMBackbone` class from MMPretrain in MMDetection, and the model used is `EfficientNet-B1`, where `mmpretrain` means the MMPretrain repo and `TIMMBackbone` means the TIMMBackbone wrapper implemented in MMPretrain.

For the principle of the Hierarchy Registry, please refer to the [MMEngine document](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md). For how to use other backbones in MMPretrain, you can refer to the [MMPretrain document](https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html).

# Use Mosaic augmentation

If you want to use `Mosaic` in training, please make sure that you use `MultiImageMixDataset` at the same time. Taking the 'Faster R-CNN' algorithm as an example, you should modify the values of `train_pipeline` and `train_dataset` in the config as below:

```python
# Open configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py directly and add the following fields
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
img_scale=(1333, 800)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)), # The image will be enlarged by 4 times after Mosaic processing,so we use affine transformation to restore the image size.
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    _delete_ = True, # remove unnecessary Settings
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    )

data = dict(
    train=train_dataset
    )
```

# Unfreeze backbone network after freezing the backbone in the config

If you have freezed the backbone network in the config and want to unfreeze it after some epoches, you can write a hook function to do it.  Taking the Faster R-CNN with the resnet backbone as an example, you can freeze one stage of the backbone network and  add a `custom_hooks` in the config as below:

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    # freeze one stage of the backbone network.
    backbone=dict(frozen_stages=1),
)
custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=1)]
```

Meanwhile write the hook class `UnfreezeBackboneEpochBasedHook` in `mmdet/core/hook/unfreeze_backbone_epoch_based_hook.py`

```python
from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class UnfreezeBackboneEpochBasedHook(Hook):
    """Unfreeze backbone network Hook.

    Args:
        unfreeze_epoch (int): The epoch unfreezing the backbone network.
    """

    def __init__(self, unfreeze_epoch=1):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            backbone = model.backbone
            if backbone.frozen_stages >= 0:
                if backbone.deep_stem:
                    backbone.stem.train()
                    for param in backbone.stem.parameters():
                        param.requires_grad = True
                else:
                    backbone.norm1.train()
                    for m in [backbone.conv1, backbone.norm1]:
                        for param in m.parameters():
                            param.requires_grad = True

            for i in range(1, backbone.frozen_stages + 1):
                m = getattr(backbone, f'layer{i}')
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
```

# Get the channels of a new backbone

If you want to get the channels of a new backbone, you can build this backbone alone and input a pseudo image to get each stage output.

Take `ResNet` as an example:

```python
from mmdet.models import ResNet
import torch
self = ResNet(depth=18)
self.eval()
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))

```

Output of the above script is as below:

```python
(1, 64, 8, 8)
(1, 128, 4, 4)
(1, 256, 2, 2)
(1, 512, 1, 1)
```

Users can get the channels of the new backbone by Replacing the `ResNet(depth=18)` in this script with their customized backbone.

# Use Detectron2 Model in MMDetection

Users can use Detectron2Wrapper to run Detectron2's model in MMDetection. We provide examples of [Faster R-CNN](../../../configs/misc/d2_faster-rcnn_r50-caffe_fpn_ms-90k_coco.py),
[Mask R-CNN](../../../configs/misc/d2_mask-rcnn_r50-caffe_fpn_ms-90k_coco.py), and [RetinaNet](../../../configs/misc/d2_retinanet_r50-caffe_fpn_ms-90k_coco.py) in MMDetection.

The algorithm components in config file should be the same as those of in Detectron2. During setup, we will first initialize the default settings, which can be found in [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py).
Then, the settings in config file will overwrite the default settings and the model will be built with these settings.
The input data will first convert to Detectron2's type and feed into Detectron2's model.
During inference the results calculate from Detectron2's model will reconvert back to the MMDetection's type.

## Use Detectron2's pre-trained weights

The weight initialization in `Detectron2Wrapper` will not use the logic of MMDetection. Users can set `model.d2_detector.weights=xxx` to load pre-trained weights.
For example, we can use `model.d2_detector.weights='detectron2://ImageNetPretrained/MSRA/R-50.pkl'` to load the pre-trained ResNet-50 or use
`model.d2_detector.weights='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl'` to load the pre-trained Mask R-CNN weights proposed in Detectron2.

**Note:** Detectron2's pretrained model cannot be loaded directly by using `load_from`, it should be first converted via `tools/model_converters/detectron2_to_mmdet.py`

For inference of released detectron2 checkpoints, users should first use `tools/model_converters/detectron2_to_mmdet.py` to convert Detectron2 checkpoint to MMDetection.

```shell
python tools/model_converters/detectron2_to_mmdet.py ${Detectron2 ckpt path} ${MMDetectron ckpt path}
```
