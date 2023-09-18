# Copyright (c) OpenMMLab. All rights reserved.

import mmengine

from mmdet.registry import DATASETS
from .coco import CocoDataset

V3DET_CLASSES = tuple(
    mmengine.list_from_file(
        'configs/v3det/category_name_13204_v3det_2023_v1.txt'))


@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Dataset for V3Det."""

    METAINFO = {
        'classes': V3DET_CLASSES,
        'palette': None,  # TODO: add palette
    }
