# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from typing import Optional

import mmengine

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Dataset for V3Det."""

    METAINFO = {
        'classes': None,
        'palette': None,
    }

    def __init__(
            self,
            *args,
            metainfo: Optional[dict] = None,
            data_root: str = '',
            label_file='annotations/category_name_13204_v3det_2023_v1.txt',  # noqa
            **kwargs) -> None:
        class_names = tuple(
            mmengine.list_from_file(os.path.join(data_root, label_file)))
        if metainfo is None:
            metainfo = {'classes': class_names}
        super().__init__(
            *args, data_root=data_root, metainfo=metainfo, **kwargs)
