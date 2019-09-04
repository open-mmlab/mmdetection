from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class HKBDataset(XMLDataset):

    CLASSES = ('1', )

    def __init__(self, **kwargs):
        super(HKBDataset, self).__init__(**kwargs)