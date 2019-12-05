from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CityscapesDataset(CocoDataset):

    CLASSES = ('person', 'rider', 'car', 'bicycle', 'bus', 'truck', 'train',
               'motorcycle')
