# 模型微调

在 COCO 数据集上预训练的检测器可以作为其他数据集（例如 CityScapes 和 KITTI 数据集）优质的预训练模型。
本教程将指导用户如何把 [ModelZoo](../model_zoo.md) 中提供的模型用于其他数据集中并使得当前所训练的模型获得更好性能。

以下是在新数据集中微调模型需要的两个步骤。

- 按 [教程2：自定义数据集](../advanced_guides/customize_dataset.md) 中的方法对新数据集添加支持中的方法对新数据集添加支持
- 按照本教程中所讨论方法，修改配置信息

接下来将会以 Cityscapes Dataset 上的微调过程作为例子，具体讲述用户需要在配置中修改的五个部分。

## 继承基础配置

为了减轻编写整个配置的负担并减少漏洞的数量， MMDetection V3.0 支持从多个现有配置中继承配置信息。微调 MaskRCNN 模型的时候，新的配置信息需要使用从 `_base_/models/mask_rcnn_r50_fpn.py` 中继承的配置信息来构建模型的基本结构。当使用 Cityscapes 数据集时，新的配置信息可以简便地从`_base_/datasets/cityscapes_instance.py` 中继承。对于训练过程的运行设置部分，例如 `logger settings`，配置文件可以从 `_base_/default_runtime.py` 中继承。对于训练计划的配置则可以从`_base_/schedules/schedule_1x.py` 中继承。这些配置文件存放于 `configs` 目录下，用户可以选择全部内容的重新编写而不是使用继承方法。

```python
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_instance.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
```

## Head 的修改

接下来新的配置还需要根据新数据集的类别数量对 Head 进行修改。只需要对 roi_head 中的 `num_classes`进行修改。修改后除了最后的预测模型的 Head 之外，预训练模型的权重的大部分都会被重新使用。

```python
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
```

## 数据集的修改

用户可能还需要准备数据集并编写有关数据集的配置，可在 [Customize Datasets](../advanced_guides/customize_dataset.md) 中获取更多信息。目前 MMDetection V3.0 的配置文件已经支持 VOC、WIDERFACE、COCO、LIVS、OpenImages、DeepFashion、Objects365 和 Cityscapes Dataset 的数据集信息。

## 训练策略的修改

微调超参数与默认的训练策略不同。它通常需要更小的学习率和更少的训练回合。

```python
# 优化器
# batch size 为 8 时的 lr 配置
optim_wrapper = dict(optimizer=dict(lr=0.01))

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[7],
        gamma=0.1)
]

# 设置 max epoch
train_cfg = dict(max_epochs=8)

# 设置 log config
default_hooks = dict(logger=dict(interval=100)),

```

## 使用预训练模型

如果要使用预训练模型，可以在 `load_from` 中查阅新的配置信息，用户需要在训练开始之前下载好需要的模型权重，从而避免在训练过程中浪费了宝贵时间。

```python
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'  # noqa
```
