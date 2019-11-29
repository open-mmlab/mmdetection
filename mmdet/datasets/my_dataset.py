from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MyDataset(CocoDataset):
    CLASSES = ('handsup', 'sleep', 'write', 'phone')
