# Migrate Configuration File from MMDetection 2.x to 3.x

The configuration file of MMDetection 3.x has undergone significant changes in comparison to the 2.x version. This document explains how to migrate 2.x configuration files to 3.x.

In the previous tutorial [Learn about Configs](../user_guides/config.md), we used Mask R-CNN as an example to introduce the configuration file structure of MMDetection 3.x. Here, we will follow the same structure to demonstrate how to migrate 2.x configuration files to 3.x.

## Model Configuration

There have been no major changes to the model configuration in 3.x compared to 2.x. For the model's backbone, neck, head, as well as train_cfg and test_cfg, the parameters remain the same as in version 2.x.

On the other hand, we have added the `DataPreprocessor` module in MMDetection 3.x. The configuration for the `DataPreprocessor` module is located in `model.data_preprocessor`. It is used to preprocess the input data, such as normalizing input images and padding images of different sizes into batches, and loading images from memory to VRAM. This configuration replaces the `Normalize` and `Pad` modules in `train_pipeline` and `test_pipeline` of the earlier version.

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
# Image normalization parameters
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
pipeline=[
    ...,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  # Padding the image to multiples of 32
    ...
]
```

</td>
<tr>
<td>2.x Config</td>
<td>

```python
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # Image normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # Image padding parameters
        pad_mask=True,  # In instance segmentation, the mask needs to be padded
        pad_size_divisor=32)  # Padding the image to multiples of 32
)

```

</td>
</tr>
</table>

## Dataset and Evaluator Configuration

The dataset and evaluator configurations have undergone major changes compared to version 2.x. We will introduce how to migrate from version 2.x to version 3.x from three aspects: Dataloader and Dataset, Data transform pipeline, and Evaluator configuration.

### Dataloader and Dataset Configuration

In the new version, we set the data loading settings consistent with PyTorch's official DataLoader,
making it easier for users to understand and get started with.
We put the data loading settings for training, validation, and testing separately in `train_dataloader`, `val_dataloader`, and `test_dataloader`.
Users can set different parameters for these dataloaders.
The input parameters are basically the same as those required by [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).

This way, we put the unconfigurable parameters in version 2.x, such as `sampler`, `batch_sampler`, and `persistent_workers`, in the configuration file, so that users can set dataloader parameters more flexibly.

Users can set the dataset configuration through `train_dataloader.dataset`, `val_dataloader.dataset`, and `test_dataloader.dataset`, which correspond to `data.train`, `data.val`, and `data.test` in version 2.x.

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,  # Avoid recreating subprocesses after each iteration
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Default batch_sampler, used to ensure that images in the batch have similar aspect ratios, so as to better utilize graphics memory
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
# In version 3.x, validation and test dataloaders can be configured independently
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here

```

</td>
</tr>
</table>

### Data Transform Pipeline Configuration

As mentioned earlier, we have separated the normalization and padding configurations for images from the `train_pipeline` and `test_pipeline`, and have placed them in `model.data_preprocessor` instead. Hence, in the 3.x version of the pipeline, we no longer require the `Normalize` and `Pad` transforms.

At the same time, we have also refactored the transform responsible for packing the data format, and have merged the `Collect` and `DefaultFormatBundle` transforms into `PackDetInputs`. This transform is responsible for packing the data from the data pipeline into the input format of the model. For more details on the input format conversion, please refer to the [data flow documentation](../advanced_guides/data_flow.md).

Below, we will use the `train_pipeline` of Mask R-CNN as an example, to demonstrate how to migrate from the 2.x configuration to the 3.x configuration:

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
```

</td>
</tr>
</table>

For the `test_pipeline`, apart from removing the `Normalize` and `Pad` transforms, we have also separated the data augmentation for testing (TTA) from the normal testing process, and have removed `MultiScaleFlipAug`. For more information on how to use the new TTA version, please refer to the [TTA documentation](../advanced_guides/tta.md).

Below, we will again use the `test_pipeline` of Mask R-CNN as an example, to demonstrate how to migrate from the 2.x configuration to the 3.x configuration:

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

</td>
</tr>
</table>

In addition, we have also refactored some data augmentation transforms. The following table lists the mapping between the transforms used in the 2.x version and the 3.x version:

<table class="docutils">
<thead>
  <tr>
    <th>Name</th>
    <th>2.x Config</th>
    <th>3.x Config</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Resize</td>
<td>

```python
dict(type='Resize',
     img_scale=(1333, 800),
     keep_ratio=True)
```

</td>

<td>

```python
dict(type='Resize',
     scale=(1333, 800),
     keep_ratio=True)
```

</td>
  </tr>
  <tr>
    <td>RandomResize</td>
<td>

```python
dict(
    type='Resize',
    img_scale=[
        (1333, 640), (1333, 800)],
    multiscale_mode='range',
    keep_ratio=True)
```

</td>
<td>

```python
dict(
    type='RandomResize',
    scale=[
        (1333, 640), (1333, 800)],
    keep_ratio=True)
```

</td>
  </tr>
  <tr>
    <td>RandomChoiceResize</td>
<td>

```python
dict(
    type='Resize',
    img_scale=[
        (1333, 640), (1333, 672),
        (1333, 704), (1333, 736),
        (1333, 768), (1333, 800)],
    multiscale_mode='value',
    keep_ratio=True)
```

</td>
<td>

```python
dict(
    type='RandomChoiceResize',
    scales=[
        (1333, 640), (1333, 672),
        (1333, 704), (1333, 736),
        (1333, 768), (1333, 800)],
    keep_ratio=True)
```

</td>
  </tr>

<tr>
    <td>RandomFlip</td>
<td>

```python
dict(type='RandomFlip', flip_ratio=0.5)
```

</td>
<td>

```python
dict(type='RandomFlip', prob=0.5)
```

</td>
  </tr>

</tbody>
</table>

### 评测器配置

In version 3.x, model accuracy evaluation is no longer tied to the dataset, but is instead accomplished through the use of an Evaluator.
The Evaluator configuration is divided into two parts: `val_evaluator` and `test_evaluator`. The `val_evaluator` is used for validation dataset evaluation, while the `test_evaluator` is used for testing dataset evaluation.
This corresponds to the `evaluation` field in version 2.x.

The following table shows the corresponding relationship between Evaluators in version 2.x and 3.x.

<table class="docutils">
<thead>
  <tr>
    <th>Metric Name</th>
    <th>2.x Config</th>
    <th>3.x Config</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>COCO</td>
<td>

```python
data = dict(
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.json'))
evaluation = dict(metric=['bbox', 'segm'])
```

</td>

<td>

```python
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False)
```

</td>
  </tr>
  <tr>
    <td>Pascal VOC</td>
<td>

```python
data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt'))
evaluation = dict(metric='mAP')
```

</td>
<td>

```python
val_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points')
```

</td>
  </tr>
  <tr>
    <td>OpenImages</td>
<td>

```python
data = dict(
    val=dict(
        type='OpenImagesDataset',
        ann_file=data_root + 'annotations/validation-annotations-bbox.csv',
        img_prefix=data_root + 'OpenImages/validation/',
        label_file=data_root + 'annotations/class-descriptions-boxable.csv',
        hierarchy_file=data_root +
        'annotations/bbox_labels_600_hierarchy.json',
        meta_file=data_root + 'annotations/validation-image-metas.pkl',
        image_level_ann_file=data_root +
        'annotations/validation-annotations-human-imagelabels-boxable.csv'))
evaluation = dict(interval=1, metric='mAP')
```

</td>
<td>

```python
val_evaluator = dict(
    type='OpenImagesMetric',
    iou_thrs=0.5,
    ioa_thrs=0.5,
    use_group_of=True,
    get_supercategory=True)
```

</td>
  </tr>

<tr>
    <td>CityScapes</td>
<td>

```python
data = dict(
    val=dict(
        type='CityScapesDataset',
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
```

</td>
<td>

```python
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        metric=['bbox', 'segm']),
    dict(
        type='CityScapesMetric',
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        seg_prefix=data_root + '/gtFine/val',
        outfile_prefix='./work_dirs/cityscapes_metric/instance')
]
```

</td>
  </tr>

</tbody>
</table>

## Configuration for Training and Testing

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
runner = dict(
    type='EpochBasedRunner',  # Type of training loop
    max_epochs=12)  # Maximum number of training epochs
evaluation = dict(interval=2)  # Interval for evaluation, check the performance every 2 epochs
```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',  # Type of training loop, please refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # Maximum number of training epochs
    val_interval=2)  # Interval for validation, check the performance every 2 epochs
val_cfg = dict(type='ValLoop')  # Type of validation loop
test_cfg = dict(type='TestLoop')  # Type of testing loop
```

</td>
</tr>
</table>

## Optimization Configuration

The configuration for optimizer and gradient clipping is moved to the `optim_wrapper` field.
The following table shows the correspondences for optimizer configuration between 2.x version and 3.x version:

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
optimizer = dict(
    type='SGD',  # Optimizer: Stochastic Gradient Descent
    lr=0.02,  # Base learning rate
    momentum=0.9,  # SGD with momentum
    weight_decay=0.0001)  # Weight decay
optimizer_config = dict(grad_clip=None)  # Configuration for gradient clipping, set to None to disable
```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
optim_wrapper = dict(  # Configuration for the optimizer wrapper
    type='OptimWrapper',  # Type of optimizer wrapper, you can switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Optimizer configuration, supports various PyTorch optimizers, please refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # SGD
        lr=0.02,  # Base learning rate
        momentum=0.9,  # SGD with momentum
        weight_decay=0.0001),  # Weight decay
    clip_grad=None,  # Configuration for gradient clipping, set to None to disable. For usage, please see https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )
```

</td>
</tr>
</table>

The configuration for learning rate is also moved from the `lr_config` field to the `param_scheduler` field. The `param_scheduler` configuration is more similar to PyTorch's learning rate scheduler and more flexible. The following table shows the correspondences for learning rate configuration between 2.x version and 3.x version:

<table class="docutils">
<tr>
<td>2.x Config</td>
<td>

```python
lr_config = dict(
    policy='step',  # Use multi-step learning rate strategy during training
    warmup='linear',  # Use linear learning rate warmup
    warmup_iters=500,  # End warmup at iteration 500
    warmup_ratio=0.001,  # Coefficient for learning rate warmup
    step=[8, 11],  # Learning rate decay at which epochs
    gamma=0.1)  # Learning rate decay coefficient

```

</td>
<tr>
<td>3.x Config</td>
<td>

```python
param_scheduler = [
    dict(
        type='LinearLR',  # Use linear learning rate warmup
        start_factor=0.001, # Coefficient for learning rate warmup
        by_epoch=False,  # Update the learning rate during warmup at each iteration
        begin=0,  # Starting from the first iteration
        end=500),  # End at the 500th iteration
    dict(
        type='MultiStepLR',  # Use multi-step learning rate strategy during training
        by_epoch=True,  # Update the learning rate at each epoch
        begin=0,   # Starting from the first epoch
        end=12,  # Ending at the 12th epoch
        milestones=[8, 11],  # Learning rate decay at which epochs
        gamma=0.1)  # Learning rate decay coefficient
]

```

</td>
</tr>
</table>

For information on how to migrate other learning rate adjustment policies, please refer to the [learning rate migration document of MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/migration/param_scheduler.html).

## Migration of Other Configurations

### Configuration for Saving Checkpoints

<table class="docutils">
<thead>
  <tr>
    <th>Function</th>
    <th>2.x Config</th>
    <th>3.x Config</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Set Save Interval</td>
<td>

```python
checkpoint_config = dict(
    interval=1)
```

</td>

<td>

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1))
```

</td>
  </tr>

<tr>
    <td>Save Best Model</td>
<td>

```python
evaluation = dict(
    save_best='auto')
```

</td>
<td>

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'))
```

</td>
  </tr>

<tr>
    <td>Keep Latest Model</td>
<td>

```python
checkpoint_config = dict(
    max_keep_ckpts=3)
```

</td>
<td>

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=3))
```

</td>
  </tr>

</tbody>
</table>

### Logging Configuration

In MMDetection 3.x, the logging and visualization of the log are carried out respectively by the logger and visualizer in MMEngine. The following table shows the comparison between the configuration of printing logs and visualizing logs in MMDetection 2.x and 3.x.

<table class="docutils">
<thead>
  <tr>
    <th>Function</th>
    <th>2.x Config</th>
    <th>3.x Config</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Set Log Printing Interval</td>
<td>

```python
log_config = dict(interval=50)
```

</td>

<td>

```python
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50))
# Optional： set moving average window size
log_processor = dict(
    type='LogProcessor', window_size=50)
```

</td>
  </tr>

<tr>
    <td>Use TensorBoard or WandB to visualize logs</td>
<td>

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'mmdetection',
                'group': 'maskrcnn-r50-fpn-1x-coco'
             },
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
    ])
```

</td>
<td>

```python
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'group': 'maskrcnn-r50-fpn-1x-coco'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```

</td>
  </tr>

</tbody>
</table>

For visualization-related tutorials, please refer to [Visualization Tutorial](../user_guides/visualization.md) of MMDetection.

### Runtime Configuration

The runtime configuration fields in version 3.x have been adjusted, and the specific correspondence is as follows:

<table class="docutils">
<thead>
  <tr>
    <th>2.x Config</th>
    <th>3.x Config</th>
  </tr>
</thead>
<tbody>

<tr>
<td>

```python
cudnn_benchmark = False
opencv_num_threads = 0
mp_start_method = 'fork'
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None


```

</td>
<td>

```python
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork',
                opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
load_from = None
resume = False
```

</td>
  </tr>

</tbody>
</table>
