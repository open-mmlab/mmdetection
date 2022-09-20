# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'CLASSES':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC2012'
        else:
            self._metainfo['DATASET_TYPE'] = None

    @property
    def metainfo(self) -> dict:
        """To use `ConcatDataset` while training, need to override this function.
        Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        metainfo = copy.deepcopy(self._metainfo)
        if 'DATASET_TYPE' in metainfo:
            del metainfo['DATASET_TYPE']
        return metainfo
