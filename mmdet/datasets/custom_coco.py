from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CustomCocoDataset(CocoDataset):

    def __init__(self, classes, *args, **kwargs):
        self.CLASSES = classes
        super().__init__(*args, **kwargs)

