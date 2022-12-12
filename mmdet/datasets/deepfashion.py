# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepFashionDataset(CocoDataset):
    """Dataset for DeepFashion."""

    METAINFO = {
        'classes': ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants',
                    'bag', 'neckwear', 'headwear', 'eyeglass', 'belt',
                    'footwear', 'hair', 'skin', 'face'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
                    (0, 192, 224), (0, 192, 192), (128, 192, 64), (0, 192, 96),
                    (128, 32, 192), (0, 0, 224), (0, 0, 64), (0, 160, 192),
                    (128, 0, 96), (128, 0, 192), (0, 32, 192)]
    }
