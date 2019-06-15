import mmcv

from .custom import CustomDataset
from .transforms import BboxTransform
import copy


class FlipVOCDataset(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.flip_transform = BboxTransform()
        self.test_mode = kwargs.get('test_mode', False)
        super(FlipVOCDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        anns = mmcv.load(ann_file)
        if self.test_mode:
            return anns
        flip_doubled_anns = []
        for ann in anns:
            ann = copy.deepcopy(ann)
            ann['flip'] = True
            flip_doubled_anns.append(ann)
            ann = copy.deepcopy(ann)
            ann['flip'] = False
            flip_doubled_anns.append(ann)
        return flip_doubled_anns
