# 将单阶段检测器作为 RPN

候选区域网络作为[Faster R-CNN](https://arxiv.org/abs/1506.01497)的一个子模块，将为Faster R-CNN的第二阶段产生候选区域。在MMDetection里大多数的二阶段检测器使用[`RPN头`](../../../mmdet/models/dense_heads/rpn_head.py)作为候选区域网络来产生候选区域。然而，任何的单阶段检测器都可以作为候选区域网络，是因为他们对边界框的预测可以被视为是一种候选区域，并且因此能够在R-CNN中得到改进。因此在MMDetection v3.0中会支持将单阶段检测器作为RPN使用。

接下来我们通过一个例子，即如何在[Faster R-CNN](../../../configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py)中使用一个无锚框的单阶段的检测器模型[FCOS](../../../configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py)作为RPN，详细阐述具体的全部流程。

主要流程如下:

1. 在Faster R-CNN 中使用`FCOS头`作为`RPN头`
2. 评估候选区域
3. 用预先训练的FCOS训练定制的Faster R-CNN

## 在Faster R-CNN中使用`FCOS头`作为`RPN头`  

为了在Faster R-CNN中使用`FCOS头`作为`RPN头`，我们应该创建一个名为`configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py`的配置文件，并且在`configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py`中将' rpn_head '的设置替换为' bbox_head '的设置，此外我们仍然使用FCOS的瓶颈设置，步幅为`[8,16,32,64,128]`，并且更新' bbox_roi_extractor '的' featmap_stride '为' [8,16,32,64,128]`。为了避免损失变慢，我们在前1000次迭代而不是前500次迭代中应用预热，这意味着lr增长得更慢。相关配置如下:

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    # copied from configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in TwoStageDetector automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(  # update featmap_strides due to the strides in neck
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])))
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),  # Slowly increase lr, otherwise loss becomes NAN
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
```

然后，我们可以使用下面的命令来训练我们的定制模型。更多训练命令，请参考 [这里](train.md)。

```python
# training with 8 GPUS
bash tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py \
    8 \
    --work-dir ./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco
```

## 评估候选区域
候选区域的质量对检测器的性能有重要影响，因此，我们也提供了一种评估候选区域的方法。和上面一样创建一个新的名为`configs/rpn/fcos-rpn_r50_fpn_1x_coco.py`的配置文件，并且在`configs/rpn/fcos-rpn_r50_fpn_1x_coco.py`中将' rpn_head '的设置替换为' bbox_head '的设置。
```python
_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator
model = dict(
    # copied from configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in RPN automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
```

假设我们在训练之后有检查点' ./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco/epoch_12.pth '，然后，我们可以使用下面的命令来评估建议的质量。

```python
# testing with 8 GPUs
bash tools/dist_test.sh \
    configs/rpn/fcos-rpn_r50_fpn_1x_coco.py \
    ./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco/epoch_12.pth \8
```

## 用预先训练的FCOS训练定制的Faster R-CNN

预训练不仅加快了训练的收敛速度，而且提高了检测器的性能。因此，我们在这里给出一个例子来说明如何使用预先训练的FCOS作为RPN来加速训练和提高精度。假设我们想在Faster R-CNN中使用“FCOSHead”作为rpn头部，并与预先训练的人员一起训练[`fcos_r50-caffe_fpn_gn-head_1x_coco`](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth)。配置文件“configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_fcos- rpn_1x_copy .py”的内容如下所示。注意，' fcos_r50-caffe_fpn_gn-head_1x_coco '使用ResNet50的caffe版本，因此需要更新' data_preprocessor '中的像素平均值和std。

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        init_cfg=None),  # the checkpoint in ``load_from`` contains the weights of backbone
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in TwoStageDetector automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(  # update featmap_strides due to the strides in neck
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])))
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
```

训练命令如下。

```python
bash tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_fcos-rpn_1x_coco.py \ 8 \
    --work-dir ./work_dirs/faster-rcnn_r50-caffe_fpn_fcos-rpn_1x_coco
```
