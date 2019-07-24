from .coco import CocoDataset


class CityscapesDataset(CocoDataset):

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle')
