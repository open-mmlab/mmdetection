# Learn about Configs

MMDetection and other OpenMMLab repositories use [MMEngine's config system](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html). It has a modular and inheritance design, which is convenient to conduct various experiments.

## Config file content

MMDetection uses a modular design, all modules with different functions can be configured through the config. Taking RTMDet as an example, we will introduce each field in the config according to different function modules:

### Model config

In MMDetection's config, we use `model` to set up detection algorithm components. In addition to neural network components such as `backbone`, `neck`, etc, it also requires `data_preprocessor`, `train_cfg`, and `test_cfg`. `data_preprocessor` is responsible for processing a batch of data output by dataloader. `train_cfg`, and `test_cfg` in the model config are for training and testing hyperparameters of the components.

```python
model = dict(
    type='RTMDet',  # The name of detector
    data_preprocessor=dict(  # The config of data preprocessor, usually includes image normalization and padding
        type='DetDataPreprocessor',  # The type of the data preprocessor. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor
        mean=[103.53, 116.28, 123.675],  # Mean values used to pre-training the pre-trained backbone models, ordered in R, G, B
        std=[57.375, 57.12, 58.395],  # Standard variance used to pre-training the pre-trained backbone models, ordered in R, G, B
        bgr_to_rgb=False,  # whether to convert image from BGR to RGB
        batch_augments=None),  # Batch-level augmentations
    backbone=dict(  # The config of backbone
        type='CSPNeXt',  # The type of backbone network. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.backbones.CSPNeXt
        arch='P5',  # Architecture of CSPNeXt, from {P5, P6}. Defaults to P5
        expand_ratio=0.5,  # Ratio to adjust the number of channels of the hidden layer. Defaults to 0.5
        deepen_factor=1,  # Depth multiplier, multiply number of blocks in CSP layer by this amount. Defaults to 1.0
        widen_factor=1,  # Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0
        channel_attention=True,  # Whether to add channel attention in each stage. Defaults to True
        norm_cfg=dict(type='SyncBN'),  # Dictionary to construct and config norm layer. Defaults to dict(type=’BN’, requires_grad=True)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Defaults to dict(type=’SiLU’)
    neck=dict(
        type='CSPNeXtPAFPN',  # The type of neck is CSPNeXtPAFPN. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.necks.CSPNeXtPAFPN
        in_channels=[256, 512, 1024],  # Number of input channels per scale
        out_channels=256,  # Number of output channels (used at each scale)
        num_csp_blocks=3,  # Number of bottlenecks in CSPLayer. Defaults to 3
        expand_ratio=0.5,  # Ratio to adjust the number of channels of the hidden layer. Default: 0.5
        norm_cfg=dict(type='SyncBN'),  # Config dict for normalization layer. Default: dict(type=’BN’)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Default: dict(type=’Swish’)
    bbox_head=dict(
        type='RTMDetSepBNHead',  # The type of bbox_head is RTMDetSepBNHead. RTMDetHead with separated BN layers and shared conv layers. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.dense_heads.RTMDetSepBNHead
        num_classes=80,  # Number of categories excluding the background category
        in_channels=256,  # Number of channels in the input feature map
        stacked_convs=2,  # Whether to share conv layers between stages. Defaults to True
        feat_channels=256,  # Feature channels of convolutional layers in the head
        anchor_generator=dict(  # The config of anchor generator
            type='MlvlPointGenerator',  # The methods use MlvlPointGenerator. Refer to https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/task_modules/prior_generators/point_generator.py#L92
            offset=0,  # The offset of points, the value is normalized with corresponding stride. Defaults to 0.5
            strides=[8, 16, 32]),  # Strides of anchors in multiple feature levels in order (w, h)
        bbox_coder=dict(type='DistancePointBBoxCoder'),  # Distance Point BBox coder.This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,right) and decode it back to the original. Refer to https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/task_modules/coders/distance_point_bbox_coder.py#L9
        loss_cls=dict(  # Config of loss function for the classification branch
            type='QualityFocalLoss',  # Type of loss for classification branch. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.losses.QualityFocalLoss
            use_sigmoid=True,  # Whether sigmoid operation is conducted in QFL. Defaults to True
            beta=2.0,  # The beta parameter for calculating the modulating factor. Defaults to 2.0
            loss_weight=1.0),  #  Loss weight of current loss
        loss_bbox=dict(  # Config of loss function for the regression branch
            type='GIoULoss',  # Type of loss. Refer to https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.models.losses.GIoULoss
            loss_weight=2.0),  # Loss weight of the regression branch
        with_objectness=False,  # Whether to add an objectness branch. Defaults to True
        exp_on_reg=True,  # Whether to use .exp() in regression
        share_conv=True,  # Whether to share conv layers between stages. Defaults to True
        pred_kernel_size=1,  # Kernel size of prediction layer. Defaults to 1
        norm_cfg=dict(type='SyncBN'),  # Config dict for normalization layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg=dict(type='SiLU', inplace=True)),  # Config dict for activation layer. Defaults to dict(type='SiLU')
    train_cfg=dict(  # Config of training hyperparameters for ATSS
        assigner=dict(  # Config of assigner
            type='DynamicSoftLabelAssigner',   # Type of assigner. DynamicSoftLabelAssigner computes matching between predictions and ground truth with dynamic soft label assignment. Refer to https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py#L40
            topk=13),  # Select top-k predictions to calculate dynamic k best matches for each gt. Defaults to 13
        allowed_border=-1,  # The border allowed after padding for valid anchors
        pos_weight=-1,  # The weight of positive samples during training
        debug=False),  # Whether to set the debug mode
    test_cfg=dict(  # Config for testing hyperparameters for ATSS
        nms_pre=30000,  # The number of boxes before NMS
        min_bbox_size=0,  # The allowed minimal box size
        score_thr=0.001,  # Threshold to filter out boxes
        nms=dict(  # Config of NMS in the second stage
            type='nms',  # Type of NMS
            iou_threshold=0.65),  # NMS threshold
        max_per_img=300),  # Max number of detections of each image
)
```

### Dataset and evaluator config

[Dataloaders](https://mmengine.readthedocs.io/en/latest/tutorials/dataset.html) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html). Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs.

```python
dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset
data_root = 'data/coco/'  # Root path of data
file_client_args = dict(backend='disk')  # file client arguments

train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile', file_client_args=file_client_args),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True),  # Whether to use bounding box, True for detection
    dict(
        type='Resize',  # Pipeline that resizes the images and their annotations
        scale=(1333, 800),  # The largest scale of the images
        keep_ratio=True  # Whether to keep the ratio between height and width
        ),
    dict(
        type='RandomFlip',  # Augmentation pipeline that flips the images and their annotations
        prob=0.5),  # The probability to flip
    dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]
test_pipeline = [  # Testing data processing pipeline
    dict(type='LoadImageFromFile', file_client_args=file_client_args),  # First pipeline to load images from file path
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # Pipeline that resizes the images
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True),  # Whether to use bounding box, True for detection
    dict(
        type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(   # Train dataloader config
    batch_size=2,  # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # training data sampler
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # randomly shuffle the training data in each epoch
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(  # Train dataset config
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # Path of annotation file
        data_prefix=dict(img='train2017/'),  # Prefix of image path
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # Config of filtering images and annotations
        pipeline=train_pipeline))
val_dataloader = dict(  # Validation dataloader config
    batch_size=1,  # Batch size of a single GPU. If batch-size > 1, the extra padding area may influence the performance.
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    drop_last=False,  # Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # not shuffle during validation and testing
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,  # Turn on the test mode of the dataset to avoid filtering annotations or images
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # Testing dataloader config
```

[Evaluators](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) are used to compute the metrics of the trained model on the validation and testing datasets. The config of evaluators consists of one or a list of metric configs:

```python
val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + 'annotations/instances_val2017.json',  # Annotation file path
    metric='bbox',  # Metrics to be evaluated, `bbox` for detection
    format_only=False)
test_evaluator = val_evaluator  # Testing evaluator config
```

Since the test dataset has no annotation files, the test_dataloader and test_evaluator config in MMDetection are generally equal to the val's. If you want to save the detection results on the test dataset, you can write the config like this:

```python
# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    metric='bbox',  # Metrics to be evaluated
    format_only=True,  # Only format and save the results to coco json file
    outfile_prefix='./work_dirs/coco_detection/test')  # The prefix of output json files
```

### Training and testing config

MMEngine's runner uses Loop to control the training, validation, and testing processes.
Users can set the maximum training epochs and validation intervals with these fields.

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type
```

### Optimization config

`optim_wrapper` is the field to configure optimization-related settings. The optimizer wrapper not only provides the functions of the optimizer, but also supports functions such as gradient clipping, mixed precision training, etc. Find more in [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html).

```python
optim_wrapper = dict(  # Optimizer wrapper config
    type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001)  # Weight decay of SGD
    )
```

`param_scheduler` is a field that configures methods of adjusting optimization hyperparameters such as learning rate and momentum. Users can combine multiple schedulers to create a desired parameter adjustment strategy. Find more in [parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html) and [parameter scheduler API documents](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.optim._ParamScheduler.html#mmengine.optim._ParamScheduler)

```python
param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=0.001, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        end=12,  # End at the 12th epoch
        milestones=[8, 11],  # Epochs to decay the learning rate
        gamma=0.1)  # The learning rate decay ratio
]
```

### Hook config

Users can attach Hooks to training, validation, and testing loops to insert some operations during running. There are two different hook fields, one is `default_hooks` and the other is `custom_hooks`.

`default_hooks` is a dict of hook configs, and they are the hooks must be required at the runtime. They have default priority which should not be modified. If not set, runner will use the default values. To disable a default hook, users can set its config to `None`. Find more in [HOOK](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html).

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    visualization=dict(type='DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results
```

`custom_hooks` is a list of all other hook configs. Users can develop their own hooks and insert them in this field.

```python
custom_hooks = []
```

### Runtime config

```python
default_scope = 'mmdet'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(  # Multi-processing config
        mp_start_method='fork',  # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend='nccl'),  # Distribution configs
)

vis_backends = [dict(type='LocalVisBackend')]  # Visualization backends. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor',  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.

log_level = 'INFO'  # The level of logging
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`
```

## Iter-based config

MMEngine's Runner also provides an iter-based training loop except for epoch-based.
To use iter-based training, users should modify the `train_cfg`, `param_scheduler`, `train_dataloader`, `default_hooks`, and `log_processor`.
Here is an example of changing an epoch-based RetinaNet config to iter-based: `configs/retinanet/retinanet_r50_fpn_90k_coco.py`

```python
# Iter-based training config
train_cfg = dict(
    _delete_=True,  # Ignore the base config setting (optional)
    type='IterBasedTrainLoop',  # Use iter-based training loop
    max_iters=90000,  # Maximum iterations
    val_interval=10000)  # Validation interval


# Change the scheduler to iter-based
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# Switch to InfiniteSampler to avoid dataloader restart
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))

# Change the checkpoint saving interval to iter-based
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))

# Change the log format to iter-based
log_processor = dict(by_epoch=False)
```

## Config file inheritance

There are 4 basic component types under `config/_base_`, dataset, model, schedule, default_runtime.
Many methods could be easily constructed with one of these models like RTMDet, Faster R-CNN, Mask R-CNN, Cascade R-CNN, RPN, SSD.
The configs that are composed by components from `_base_` are called the _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made based on Faster R-CNN, users may first inherit the basic Faster R-CNN structure by specifying `_base_ = ../rtmdet/rtmdet_l_8xb32-300e_coco.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `RTMDet_xxx` under `configs`,

Please refer to [mmengine config tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) for detailed documentation.

By setting the `_base_` field, we can set which files the current configuration file inherits from.

When `_base_` is a string of a file path, it means inheriting the contents from one config file.

```python
_base_ = './rtmdet_l_8xb32-300e_coco.py'
```

When `_base_` is a list of multiple file paths, it means inheriting from multiple files.

```python
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py',
    './rtmdet_tta.py'
]
```

If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of the fields in base configs.
You may refer to [mmengine config tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) for a simple illustration.

In MMDetection, for example, to change the backbone of Mask R-CNN with the following config.

```python
model = dict(
    type='RTMDet',
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(...),
    rpn_head=dict(...),
    roi_head=dict(...))
```

`CSPNeXt` and `HRNet` use different keywords to construct.

```python
_base_ = '../rtmdet/rtmdet_l_8xb32-300e_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')),
    neck=dict(...))
```

The `_delete_=True` would replace all old keys in `backbone` field with new keys.

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.
For example, we would like to use a multi-scale strategy to train a Mask R-CNN. `train_pipeline`/`test_pipeline` are intermediate variables we would like to modify.

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

We first define the new `train_pipeline`/`test_pipeline` and pass them into dataloader fields.

Similarly, if we would like to switch from `SyncBN` to `BN` or `MMSyncBN`, we need to substitute every `norm_cfg` in the config.

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```

### Reuse variables in \_base\_ file

If the users want to reuse the variables in the base file, they can get a copy of the corresponding variable by using `{{_base_.xxx}}`. E.g:

```python
_base_ = './rtmdet_l_8xb32-300e_coco.py'

a = {{_base_.model}} # Variable `a` is equal to the `model` defined in `_base_`
```

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_dataloader.dataset.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromNDArray'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromNDArray`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`. If you want to
  change the mean values, you may specify `--cfg-options model.data_preprocessor.mean="[127,127,127]"`. Note that the quotation mark `"` is necessary to
  support list/tuple data types, and **NO** white space is allowed inside the quotation marks in the specified value.

## Config name style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

The file name is divided into five parts. All parts and components are connected with `_` and words of each part or component should be connected with `-`.

- `{algorithm name}`: The name of the algorithm. It can be a detector name such as `faster-rcnn`, `mask-rcnn`, etc. Or can be a semi-supervised or knowledge-distillation algorithm such as `soft-teacher`, `lad`. etc.
- `{model component names}`: Names of the components used in the algorithm such as backbone, neck, etc. For example, `r50-caffe_fpn_gn-head` means using caffe-style ResNet50, FPN and detection head with Group Norm in the algorithm.
- `{training settings}`: Information of training settings such as batch size, augmentations, loss trick, scheduler, and epochs/iterations. For example: `4xb4-mixup-giou-coslr-100e` means using 8-gpus x 4-images-per-gpu, mixup augmentation, GIoU loss, cosine annealing learning rate, and train 100 epochs.
  Some abbreviations:
  - `{gpu x batch_per_gpu}`: GPUs and samples per GPU. `bN` indicates N batch size per GPU. E.g. `4xb4` is the short term of 4-GPUs x 4-images-per-GPU. And `8xb2` is used by default if not mentioned.
  - `{schedule}`: training schedule, options are `1x`, `2x`, `20e`, etc.
    `1x` and `2x` means 12 epochs and 24 epochs respectively.
    `20e` is adopted in cascade models, which denotes 20 epochs.
    For `1x`/`2x`, the initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs.
    For `20e`, the initial learning rate decays by a factor of 10 at the 16th and 19th epochs.
- `{training dataset information}`: Training dataset names like `coco`, `coco-panoptic`, `cityscapes`, `voc-0712`, `wider-face`.
- `{testing dataset information}` (optional): Testing dataset name for models trained on one dataset but tested on another. If not mentioned, it means the model was trained and tested on the same dataset type.
