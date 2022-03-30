# 3: 在标准数据集上训练自定义模型
在本文中，你将知道如何在标准数据集上训练、测试和推理自定义模型。我们将在 cityscapes 数据集上以自定义 Cascade Mask R-CNN R50 模型为例演示整个过程，为了方便说明，我们将 neck 模块中的 `FPN` 替换为 `AugFPN`，并且在训练中的自动增强类中增加 `Rotate` 或 `Translate`。

基本步骤如下所示：

1. 准备标准数据集
2. 准备你的自定义模型
3. 准备配置文件
4. 在标准数据集上对模型进行训练、测试和推理

## 准备标准数据集

在本文中，我们使用 cityscapes 标准数据集为例进行说明。

推荐将数据集根路径采用符号链接方式链接到 `$MMDETECTION/data`。

如果你的文件结构不同，你可能需要在配置文件中进行相应的路径更改。标准的文件组织格式如下所示：

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

你也可以通过如下方式设定数据集根路径
```bash
export MMDET_DATASETS=$data_root
```
我们将会使用环境便变量 `$MMDET_DATASETS` 作为数据集的根目录，因此你无需再修改相应配置文件的路径信息。


你需要使用脚本 `tools/dataset_converters/cityscapes.py` 将 cityscapes 标注转化为 coco 标注格式。

```shell
pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py ./data/cityscapes --nproc 8 --out-dir ./data/cityscapes/annotations
```

目前在 `cityscapes `文件夹中的配置文件所对应模型是采用 COCO 预训练权重进行初始化的。

如果你的网络不可用或者比较慢，建议你先手动下载对应的预训练权重，否则可能在训练开始时候出现错误。

## 准备你的自定义模型

第二步是准备你的自定义模型或者训练相关配置。假设你想在已有的  Cascade Mask R-CNN R50 检测模型基础上，新增一个新的 neck 模块 `AugFPN` 去代替默认的 `FPN`，以下是具体实现：

### 1 定义新的 neck (例如 AugFPN)

首先创建新文件  `mmdet/models/necks/augfpn.py`.

```python
from ..builder import NECKS

@NECKS.register_module()
class AugFPN(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

### 2 导入模块

你可以采用两种方式导入模块，第一种是在  `mmdet/models/necks/__init__.py` 中添加如下内容

```python
from .augfpn import AugFPN
```

第二种是增加如下代码到对应配置中，这种方式的好处是不需要改动代码

```python
custom_imports = dict(
    imports=['mmdet.models.necks.augfpn.py'],
    allow_failed_imports=False)
```

### 3 修改配置

```python
neck=dict(
    type='AugFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

关于自定义模型其余相关细节例如实现新的骨架网络，头部网络、损失函数，以及运行时训练配置例如定义新的优化器、使用梯度裁剪、定制训练调度策略和钩子等，请参考文档 [自定义模型](tutorials/customize_models.md) 和 [自定义运行时训练配置](tutorials/customize_runtime.md)。

## 准备配置文件

第三步是准备训练配置所需要的配置文件。假设你打算基于 cityscapes 数据集，在 Cascade Mask R-CNN R50 中新增 `AugFPN` 模块，同时增加 `Rotate` 或者 `Translate` 数据增强策略，假设你的配置文件位于 `configs/cityscapes/` 目录下，并且取名为 `cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py`，则配置信息如下：

```python
# 继承 base 配置，然后进行针对性修改
_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_instance.py', '../_base_/default_runtime.py'
]

model = dict(
    # 设置为 None，表示不加载 ImageNet 预训练权重，
    # 后续可以设置 `load_from` 参数用来加载 COCO 预训练权重
    backbone=dict(init_cfg=None),
    pretrained=None,
    # 使用新增的 `AugFPN` 模块代替默认的 `FPN`
    neck=dict(
        type='AugFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    # 我们也需要将 num_classes 从 80 修改为 8 来匹配 cityscapes 数据集标注
    # 这个修改包括 `bbox_head` 和 `mask_head`.
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # 将 COCO 类别修改为 cityscapes 类别
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # 将 COCO 类别修改为 cityscapes 类别
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # 将 COCO 类别修改为 cityscapes 类别
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            # 将 COCO 类别修改为 cityscapes 类别
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))

# 覆写 `train_pipeline`，然后新增 `AutoAugment` 训练配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='AutoAugment',
        policies=[
            [dict(
                 type='Rotate',
                 level=5,
                 img_fill_val=(124, 116, 104),
                 prob=0.5,
                 scale=1)
            ],
            [dict(type='Rotate', level=7, img_fill_val=(124, 116, 104)),
             dict(
                 type='Translate',
                 level=5,
                 prob=0.5,
                 img_fill_val=(124, 116, 104))
            ],
        ]),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# 设置每张显卡的批处理大小，同时设置新的训练 pipeline
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3,
    # 用新的训练 pipeline 配置覆写 pipeline
    train=dict(dataset=dict(pipeline=train_pipeline)))

# 设置优化器
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 设置定制的学习率策略
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=10)

# 我们采用 COCO 预训练过的 Cascade Mask R-CNN R50 模型权重作为初始化权重，可以得到更加稳定的性能
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
```

## 训练新模型

为了能够使用新增配置来训练模型，你可以运行如下命令：

```shell
python tools/train.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py
```

如果想了解更多用法，可以参考 [例子1](1_exist_data_model.md)。

## 测试和推理

为了能够测试训练好的模型，你可以运行如下命令：

```shell
python tools/test.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py work_dirs/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py/latest.pth --eval bbox segm
```

如果想了解更多用法，可以参考 [例子1](1_exist_data_model.md)。
