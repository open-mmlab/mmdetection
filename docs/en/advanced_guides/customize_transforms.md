# Customize Data Pipelines

1. Write a new transform in a file, e.g., in `my_pipeline.py`. It takes a dict as input and returns a dict.

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

2. Import and use the pipeline in your config file.
   Make sure the import is relative to where your train script is located.

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

3. Visualize the output of your transforms pipeline

   To visualize the output of your transforms pipeline, `tools/misc/browse_dataset.py`
   can help the user to browse a detection dataset (both images and bounding box annotations)
   visually, or save the image to a designated directory. More details can refer to
   [visualization documentation](../user_guides/visualization.md)
