# 自定义数据预处理流程

1. 在任意文件里写一个新的流程，例如在 `my_pipeline.py`，它以一个字典作为输入并且输出一个字典：

   ```python
   import random
   from mmcv.transforms import BaseTransform
   from mmdet.registry import TRANSFORMS


   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):
       """Add your transform

       Args:
           p (float): Probability of shifts. Default 0.5.
       """

       def __init__(self, prob=0.5):
           self.prob = prob

       def transform(self, results):
           if random.random() > self.prob:
               results['dummy'] = True
           return results
   ```

2. 在配置文件里调用并使用你写的数据处理流程，需要确保你的训练脚本能够正确导入新增模块：

   ```python
   custom_imports = dict(imports=['path.to.my_pipeline'], allow_failed_imports=False)

   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations', with_bbox=True),
       dict(type='Resize', scale=(1333, 800), keep_ratio=True),
       dict(type='RandomFlip', prob=0.5),
       dict(type='MyTransform', prob=0.2),
       dict(type='PackDetInputs')
   ]
   ```

3. 可视化数据增强处理流程的结果

   如果想要可视化数据增强处理流程的结果，可以使用 `tools/misc/browse_dataset.py` 直观
   地浏览检测数据集（图像和标注信息），或将图像保存到指定目录。
   使用方法请参考[可视化文档](../user_guides/visualization.md)
