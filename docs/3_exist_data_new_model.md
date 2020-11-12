# 3: Inference and training with existing models and standard datasets

In this note, you will know how to inference, test, and train predefined models with your own settings for standard datasets. We use the cityscapes dataset to train a COCO pretrained Cascade Mask R-CNN model as an example to describe the whole process.

The basic steps are as below:

1. Prepare the standard dataset
2. Prepare a config
3. Train, test, inference models on the standard dataset.

### Prepare the standard dataset

In this note, as we use the standard cityscapes dataset as an example.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
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

The cityscapes annotations have to be converted into the coco format using `tools/convert_datasets/cityscapes.py`:

```shell
pip install cityscapesscripts
python tools/convert_datasets/cityscapes.py ./data/cityscapes --nproc 8 --out-dir ./data/cityscapes/annotations
```

Currently the config files in `cityscapes` use COCO pre-trained weights to initialize.
You could download the pre-trained models in advance if network is unavailable or slow, otherwise it would cause errors at the beginning of training.

### Prepare a config

The second step is to prepare a config for your own training setting. Assume that we want to use Cascade Mask R-CNN with FPN to train the cityscapes dataset, and assume the config is under directory `configs/cityscapes/` and named as `cascade_mask_rcnn_r50_fpn_1x_cityscapes.py`, the config is as below.

```python
# The new config inherits the base configs to highlight the necessary modification
_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_instance.py', '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head from 80 to 8, to match the cityscapes dataset's annotation.
# This modification involves `bbox_head` and `mask_head`.
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
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
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))

# Set optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# Set learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
total_epochs = 8

# We can use the COCO pretrained Cascade Mask R-CNN model for more stable performance initialization
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
```

### Train a new model

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/cityscapes/cascade_mask_rcnn_r50_fpn_1x_cityscapes.py
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).

### Test and inference

To test the trained model, you can simply run

```shell
python tools/test.py configs/cityscapes/cascade_mask_rcnn_r50_fpn_1x_cityscapes.py work_dirs/cascade_mask_rcnn_r50_fpn_1x_cityscapes/latest.pth --eval bbox segm
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).
