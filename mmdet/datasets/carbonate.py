from .coco import CocoDataset


class CarbonateDataset(CocoDataset):
    CLASSES = ('grain', 'cement')
