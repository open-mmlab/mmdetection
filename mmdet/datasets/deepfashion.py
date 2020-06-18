from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepFashionDataset(CocoDataset):

    CLASSES = ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
               'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
               'skin', 'face')
