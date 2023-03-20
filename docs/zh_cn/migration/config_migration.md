# 将配置文件从 MMDetection 2.x 迁移至 3.x

MMDetection 3.x 的配置文件与 2.x 相比有较大变化，这篇文档将介绍如何将 2.x 的配置文件迁移到 3.x。

在前面的[配置文件教程](../user_guides/config.md)中，我们以 Mask R-CNN 为例介绍了 MMDetection 3.x 的配置文件结构，这里我们将按同样的结构介绍如何将 2.x 的配置文件迁移至 3.x。

## 模型配置

模型的配置与 2.x 相比并没有太大变化，对于模型的 backbone，neck，head，以及 train_cfg 和 test_cfg，它们的参数与 2.x 版本的参数保持一致。

不同的是，我们在 3.x 版本的模型中新增了 `DataPreprocessor` 模块。
`DataPreprocessor` 模块的配置位于 `model.data_preprocessor` 中，它用于对输入数据进行预处理，例如对输入图像进行归一化，将不同大小的图片进行 padding 从而组成 batch，将图像从内存中读取到显存中等。这部分配置取代了原本存在于 train_pipeline 和 test_pipeline 中的 `Normalize` 和 `Pad`。

<table class="docutils">
<tr>
<td>原配置</td>
<td>

```python
# 图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
pipeline=[
    ...,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  # 图像 padding 到 32 的倍数
    ...
]
```

</td>
<tr>
<td>新配置</td>
<td>

```python
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # 图像归一化参数
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # 图像 padding 参数
        pad_mask=True,  # 在实例分割中，需要将 mask 也进行 padding
        pad_size_divisor=32)  # 图像 padding 到 32 的倍数
)
```

</td>
</tr>
</table>

## 数据集和评测器配置

数据集和评测部分的配置相比 2.x 版本有较大的变化。我们将从 Dataloader 和 Dataset，Data transform pipeline，以及评测器配置三个方面介绍如何将 2.x 版本的配置迁移到 3.x 版本。

### Dataloader 和 Dataset 配置

在新版本中，我们将数据加载的设置与 PyTorch 官方的 DataLoader 保持一致，这样可以使用户更容易理解和上手。
我们将训练、验证和测试的数据加载设置分别放在 `train_dataloader`，`val_dataloader` 和 `test_dataloader` 中，用户可以分别对这些 dataloader 设置不同的参数，其输入参数与 [PyTorch 的 Dataloader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) 所需要的参数基本一致。

通过这种方式，我们将 2.x 版本中不可配置的 `sampler`，`batch_sampler`，`persistent_workers` 等参数都放到了配置文件中，使得用户可以更加灵活地设置数据加载的参数。

用户可以通过 `train_dataloader.dataset`，`val_dataloader.dataset` 和 `test_dataloader.dataset` 来设置数据集的配置，它们分别对应 2.x 版本中的 `data.train`，`data.val` 和 `data.test`。

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
<td>

```python
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,  # 避免每次迭代后 dataloader 重新创建子进程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 默认的 sampler，同时支持分布式训练和非分布式训练
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 默认的 batch_sampler，用于保证 batch 中的图片具有相似的长宽比，从而可以更好地利用显存
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
# 在 3.x 版本中可以独立配置验证和测试的 dataloader
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
test_dataloader = val_dataloader  # 测试 dataloader 的配置与验证 dataloader 的配置相同，这里省略
```

</td>
</tr>
</table>

### Data transform pipeline 配置

上文中提到，我们将图像 normalize 和 padding 的配置从 `train_pipeline` 和 `test_pipeline` 中独立出来，放到了 `model.data_preprocessor` 中，因此在 3.x 版本的 pipeline 中，我们不再需要 `Normalize` 和 `Pad` 这两个 transform。

同时，我们也对负责数据格式打包的 transform 进行了重构，将 `Collect` 和 `DefaultFormatBundle` 这两个 transform 合并为了 `PackDetInputs`，它负责将 data pipeline 中的数据打包成模型的输入格式，关于输入格式的转换，详见[数据流文档](../advanced_guides/data_flow.md)。

下面以 Mask R-CNN 1x 的 train_pipeline 为例，介绍如何将 2.x 版本的配置迁移到 3.x 版本：

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

对于 test_pipeline，除了将 `Normalize` 和 `Pad` 这两个 transform 去掉之外，我们也将测试时的数据增强（TTA）与普通的测试流程分开，移除了 `MultiScaleFlipAug`。关于新版的 TTA 如何使用，详见[TTA 文档](../advanced_guides/tta.md)。

下面同样以 Mask R-CNN 1x 的 test_pipeline 为例，介绍如何将 2.x 版本的配置迁移到 3.x 版本：

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

除此之外，我们还对一些数据增强进行了重构，下表列出了 2.x 版本中的 transform 与 3.x 版本中的 transform 的对应关系：

<table class="docutils">
<thead>
  <tr>
    <th>名称</th>
    <th>原配置</th>
    <th>新配置</th>
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
dict(type='RandomFlip',
     flip_ratio=0.5)
```

</td>
<td>

```python
dict(type='RandomFlip',
     prob=0.5)
```

</td>
  </tr>

</tbody>
</table>

### 评测器配置

在 3.x 版本中，模型精度评测不再与数据集绑定，而是通过评测器（Evaluator）来完成。
评测器配置分为 val_evaluator 和 test_evaluator 两部分，其中 val_evaluator 用于验证集评测，test_evaluator 用于测试集评测，对应 2.x 版本中的 evaluation 字段。
下表列出了 2.x 版本与 3.x 版本中的评测器的对应关系：

<table class="docutils">
<thead>
  <tr>
    <th>评测指标名称</th>
    <th>原配置</th>
    <th>新配置</th>
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

## 训练和测试的配置

<table class="docutils">
<tr>
<td>原配置</td>
<td>

```python
runner = dict(
    type='EpochBasedRunner',  # 训练循环的类型
    max_epochs=12)  # 最大训练轮次
evaluation = dict(interval=2)  # 验证间隔。每 2 个 epoch 验证一次
```

</td>
<tr>
<td>新配置</td>
<td>

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # 最大训练轮次
    val_interval=2)  # 验证间隔。每 2 个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型
```

</td>
</tr>
</table>

## 优化相关配置

优化器以及梯度裁剪的配置都移至 optim_wrapper 字段中。下表列出了 2.x 版本与 3.x 版本中的优化器配置的对应关系：

<table class="docutils">
<tr>
<td>原配置</td>
<td>

```python
optimizer = dict(
    type='SGD',  # 随机梯度下降优化器
    lr=0.02,  # 基础学习率
    momentum=0.9,  # 带动量的随机梯度下降
    weight_decay=0.0001)  # 权重衰减
optimizer_config = dict(grad_clip=None)  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪
```

</td>
<tr>
<td>新配置</td>
<td>

```python
optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.02,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )
```

</td>
</tr>
</table>

学习率的配置也从 lr_config 字段中移至 param_scheduler 字段中。param_scheduler 的配置更贴近 PyTorch 的学习率调整策略，更加灵活。下表列出了 2.x 版本与 3.x 版本中的学习率配置的对应关系：

<table class="docutils">
<tr>
<td>原配置</td>
<td>

```python
lr_config = dict(
    policy='step',  # 在训练过程中使用 multi step 学习率策略
    warmup='linear',  # 使用线性学习率预热
    warmup_iters=500,  # 到第 500 个 iteration 结束预热
    warmup_ratio=0.001,  # 学习率预热的系数
    step=[8, 11],  # 在哪几个 epoch 进行学习率衰减
    gamma=0.1)  # 学习率衰减系数
```

</td>
<tr>
<td>新配置</td>
<td>

```python
param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]
```

</td>
</tr>
</table>

关于其他的学习率调整策略的迁移，请参考 MMEngine 的[学习率迁移文档](https://mmengine.readthedocs.io/zh_CN/latest/migration/param_scheduler.html)。

## 其他配置的迁移

### 保存 checkpoint 的配置

<table class="docutils">
<thead>
  <tr>
    <th>功能</th>
    <th>原配置</th>
    <th>新配置</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>设置保存间隔</td>
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
    <td>保存最佳模型</td>
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
    <td>只保留最新的几个模型</td>
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

### 日志的配置

3.x 版本中，日志的打印和可视化由 MMEngine 中的 logger 和 visualizer 分别完成。下表列出了 2.x 版本与 3.x 版本中的日志配置的对应关系：

<table class="docutils">
<thead>
  <tr>
    <th>功能</th>
    <th>原配置</th>
    <th>新配置</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>设置日志打印间隔</td>
<td>

```python
log_config = dict(
    interval=50)
```

</td>

<td>

```python
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50))
# 可选： 配置日志打印数值的平滑窗口大小
log_processor = dict(
    type='LogProcessor',
    window_size=50)
```

</td>
  </tr>

<tr>
    <td>使用 TensorBoard 或 WandB 可视化日志</td>
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
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

</td>
  </tr>

</tbody>
</table>

关于可视化相关的教程，请参考 MMDetection 的[可视化教程](../user_guides/visualization.md)。

### Runtime 的配置

3.x 版本中 runtime 的配置字段有所调整，具体的对应关系如下：

<table class="docutils">
<thead>
  <tr>
    <th>原配置</th>
    <th>新配置</th>
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
