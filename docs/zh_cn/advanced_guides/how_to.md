本教程收集了任何如何使用 MMDetection 进行 xxx 的答案。 如果您遇到有关`如何做`的问题及答案，请随时更新此文档！

## 使用 MMClassification 的骨干网络

MMDet、MMCls、MMSeg 中的模型注册表都继承自 MMEngine 中的根注册表，允许这些存储库直接使用彼此已经实现的模块。 因此用户可以在 MMDetection 中使用来自 MMClassification 的骨干网络，而无需实现MMClassification 中已经存在的网络。

### 使用在 MMClassification 中实现的骨干网络

假设想将 `MobileNetV3-small` 作为 `RetinaNet` 的骨干网络，则配置文件如下。

```python
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# please install mmcls>=1.0.0rc0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
pretrained = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'
model = dict(
    backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmcls.MobileNetV3', # 使用 mmcls 中的 MobileNetV3
        arch='small',
        out_indices=(3, 8, 11), # 修改 out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')), # MMCls 中骨干网络的预训练权重含义 prefix='backbone.'，为了正常加载权重，需要把这个 prefix 去掉。
    # 修改 in_channels
    neck=dict(in_channels=[24, 48, 96], start_level=0))
```

### 通过 MMClassification 使用 TIMM 中实现的骨干网络

由于 MMClassification 提供了 Py**T**orch **Im**age **M**odels (`timm`) 骨干网络的封装，用户也可以通过 MMClassification 直接使用 `timm` 中的骨干网络。假设想将 [`EfficientNet-B1`](../../../configs/timm_example/retinanet_timm-efficientnet-b1_fpn_1x_coco.py) 作为 `RetinaNet` 的骨干网络，则配置文件如下。

```python
# https://github.com/open-mmlab/mmdetection/blob/main/configs/timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco.py
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# please install mmcls>=1.0.0rc0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmcls.TIMMBackbone', # 使用 mmcls 中 timm 骨干网络
        model_name='efficientnet_b1',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)), # 修改 out_indices
    neck=dict(in_channels=[24, 40, 112, 320])) # 修改 in_channels

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

`type='mmcls.TIMMBackbone'` 表示在 MMDetection 中使用 MMClassification 中的 `TIMMBackbone` 类，并且使用的模型为` EfficientNet-B1`，其中 `mmcls` 表示 MMClassification 库，而 `TIMMBackbone ` 表示 MMClassification 中实现的 TIMMBackbone 包装器。

关于层次注册器的具体原理可以参考 [MMEngine 文档](https://mmengine.readthedocs.io/zh_cn/latest/tutorials/config.md#跨项目继承配置文件)，关于如何使用 MMClassification 中的其他 backbone，可以参考 [MMClassification 文档](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/tutorials/config.md)。

## 使用马赛克数据增强

如果你想在训练中使用 `Mosaic`，那么请确保你同时使用 `MultiImageMixDataset`。以 `Faster R-CNN` 算法为例，你可以通过如下做法实现：

```python
# 直接打开 configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py ,增添如下字段
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
img_scale=(1333, 800)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)), # 图像经过马赛克处理后会放大4倍，所以我们使用仿射变换来恢复图像的大小。
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'))
]

train_dataset = dict(
    _delete_ = True, # 删除不必要的设置
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

## 在配置文件中冻结骨干网络后在训练中解冻骨干网络

如果你在配置文件中已经冻结了骨干网络并希望在几个训练周期后解冻它，你可以通过 hook 来实现这个功能。以用 ResNet 为骨干网络的 Faster R-CNN 为例，你可以冻结一个骨干网络的一个层并在配置文件中添加如下 `custom_hooks`:

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

同时在 `mmdet/core/hook/unfreeze_backbone_epoch_based_hook.py` 当中书写 `UnfreezeBackboneEpochBasedHook` 类

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
            if is_module_wrapper(model):
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

## 获得新的骨干网络的通道数

如果你想获得一个新骨干网络的通道数，你可以单独构建这个骨干网络并输入一个伪造的图片来获取每一个阶段的输出。

以 `ResNet` 为例：

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

以上脚本的输出为:

```python
(1, 64, 8, 8)
(1, 128, 4, 4)
(1, 256, 2, 2)
(1, 512, 1, 1)
```

用户可以通过将脚本中的 `ResNet(depth=18)` 替换为自己的骨干网络配置来得到新的骨干网络的通道数。

# MMDetection 中训练 Detectron2 的模型

用户可以使用 `Detectron2Wrapper` 从而在 MMDetection 中使用 Detectron2 的模型。
我们提供了 [Faster R-CNN](../../../configs/misc/d2_faster-rcnn_r50-caffe_fpn_ms-90k_coco.py),
[Mask R-CNN](../../../configs/misc/d2_mask-rcnn_r50-caffe_fpn_ms-90k_coco.py) 和 [RetinaNet](../../../configs/misc/d2_retinanet_r50-caffe_fpn_ms-90k_coco.py) 的示例来在 MMDetection 中训练/测试 Detectron2 的模型。

使用过程中需要注意配置文件中算法组件要和 Detectron2 中的相同。模型初始化时，我们首先初始化 [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py) 的默认设置，然后配置文件中的设置将覆盖默认设置，模型将基于更新过的设置来建立。
输入数据首先转换成 Detectron2 的类型并输入进 Detectron2 的模型中。在推理阶段，Detectron2 的模型结果将会转换回 MMDetection 的类型。

## 使用 Detectron2 的预训练权重

`Detectron2Wrapper` 中的权重初始化将不使用 MMDetection 的逻辑。用户可以设置 `model.d2_detector.weights=xxx` 来加载预训练的权重。
例如，我们可以使用 `model.d2_detector.weights='detectron2://ImageNetPretrained/MSRA/R-50.pkl'` 来加载 ResNet-50 的预训练权重，或者使用
`model.d2_detector.weights='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl'` 来加载 Detectron2 中提出的预训练的Mask R-CNN权重。

**注意：** 不能直接使用 `load_from` 来加载 Detectron2 的预训练模型，但可以通过 `tools/model_converters/detectron2_to_mmdet.py` 先对该预训练模型进行转换。

在测试时，用户应该首先使用 `tools/model_converters/detectron2_to_mmdet.py` 将 Detectron2 的预训练权重转换为 MMDetection 可读取的结构。

```shell
python tools/model_converters/detectron2_to_mmdet.py ${Detectron2 ckpt path} ${MMDetectron ckpt path}。
```
