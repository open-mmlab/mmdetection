from .registry import DATASETS
from .xml_style import XMLDataset
from .coco import CocoDataset


@DATASETS.register_module
class KI67Dataset(XMLDataset):

    CLASSES = ('positive', 'negative', 'stromal', 'lymphocyte')

    def __init__(self, **kwargs):
        super(KI67Dataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')


@DATASETS.register_module
class KI67MaskDataset(CocoDataset):
    CLASSES = ('positive', 'negative', 'stromal', 'lymphocyte')
