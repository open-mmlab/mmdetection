# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset
from .coco import CocoDataset

import mmengine

V3DET_CLASSES = tuple(
    mmengine.list_from_file(
        'data/V3Det/annotations/category_name_13204_v3det_2023_v1.txt'))


@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Dataset for V3Det."""

    METAINFO = {
        'classes': V3DET_CLASSES,
        'palette': None,  # TODO: add palette
    }
