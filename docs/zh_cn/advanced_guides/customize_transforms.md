# 自定义数据预处理流程（待更新）

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
