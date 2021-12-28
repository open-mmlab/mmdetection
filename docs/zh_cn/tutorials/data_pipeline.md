# 教程 3: 自定义数据预处理流程

## 数据流程的设计

按照惯例，我们使用 `Dataset` 和 `DataLoader` 进行多进程的数据加载。`Dataset` 返回字典类型的数据，数据内容为模型 `forward` 方法的各个参数。由于在目标检测中，输入的图像数据具有不同的大小，我们在 `MMCV` 里引入一个新的 `DataContainer` 类去收集和分发不同大小的输入数据。更多细节请参考[这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)。

数据的准备流程和数据集是解耦的。通常一个数据集定义了如何处理标注数据（annotations）信息，而一个数据流程定义了准备一个数据字典的所有步骤。一个流程包括一系列的操作，每个操作都把一个字典作为输入，然后再输出一个新的字典给下一个变换操作。

我们在下图展示了一个经典的数据处理流程。蓝色块是数据处理操作，随着数据流程的处理，每个操作都可以在结果字典中加入新的键（标记为绿色）或更新现有的键（标记为橙色）。

![pipeline figure](../../../resources/data_pipeline.png)

这些操作可以分为数据加载（data loading）、预处理（pre-processing）、格式变化（formatting）和测试时数据增强（test-time augmentation）。

下面的例子是 `Faster R-CNN` 的一个流程：

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

对于每个操作，我们列出它添加、更新、移除的相关字典域 (dict fields)：

### 数据加载 Data loading

`LoadImageFromFile`

- 增加：img, img_shape, ori_shape

`LoadAnnotations`

- 增加：gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg, bbox_fields, mask_fields

`LoadProposals`

- 增加：proposals

### 预处理 Pre-processing

`Resize`

- 增加：scale, scale_idx, pad_shape, scale_factor, keep_ratio
- 更新：img, img_shape, *bbox_fields, *mask_fields, *seg_fields

`RandomFlip`

- 增加：flip
- 更新：img, *bbox_fields, *mask_fields, *seg_fields

`Pad`

- 增加：pad_fixed_size, pad_size_divisor
- 更新：img, pad_shape, *mask_fields, *seg_fields

`RandomCrop`

- 更新：img, pad_shape, gt_bboxes, gt_labels, gt_masks, *bbox_fields

`Normalize`

- 增加：img_norm_cfg
- 更新：img

`SegRescale`

- 更新：gt_semantic_seg

`PhotoMetricDistortion`

- 更新：img

`Expand`

- 更新：img, gt_bboxes

`MinIoURandomCrop`

- 更新：img, gt_bboxes, gt_labels

`Corrupt`

- 更新：img

### 格式 Formatting

`ToTensor`

- 更新：由 `keys` 指定

`ImageToTensor`

- 更新：由 `keys` 指定

`Transpose`

- 更新：由 `keys` 指定

`ToDataContainer`

- 更新：由 `keys` 指定

`DefaultFormatBundle`

- 更新：img, proposals, gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg

`Collect`

- 增加：img_metas（img_metas 的键（key）被 `meta_keys` 指定)
- 移除：除了 `keys` 指定的键（key）之外的所有其他的键（key）

### 测试时数据增强 Test time augmentation

`MultiScaleFlipAug`

## 拓展和使用自定义的流程

1. 在任意文件里写一个新的流程，例如在 `my_pipeline.py`，它以一个字典作为输入并且输出一个字典：

    ```python
    import random
    from mmdet.datasets import PIPELINES


    @PIPELINES.register_module()
    class MyTransform:
        """Add your transform

        Args:
            p (float): Probability of shifts. Default 0.5.
        """

        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, results):
            if random.random() > self.p:
                results['dummy'] = True
            return results
    ```

2. 在配置文件里调用并使用你写的数据处理流程，需要确保你的训练脚本能够正确导入新增模块：

    ```python
    custom_imports = dict(imports=['path.to.my_pipeline'], allow_failed_imports=False)

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='MyTransform', p=0.2),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
    ```

3. 可视化数据增强处理流程的结果

   如果想要可视化数据增强处理流程的结果，可以使用 `tools/misc/browse_dataset.py` 直观
   地浏览检测数据集（图像和标注信息），或将图像保存到指定目录。
   使用方法请参考[日志分析](../useful_tools.md)
