import pytest

from mmdet.datasets import DATASETS


def test_xml_dataset():
    dataconfig = {
        'ann_file': 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        'img_prefix': 'data/VOCdevkit/VOC2007/',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }]
    }
    XMLDataset = DATASETS.get('XMLDataset')

    class XMLDatasetSubClass(XMLDataset):
        CLASSES = None

    # get_ann_info and _filter_imgs of XMLDataset
    # would use self.CLASSES, we added CLASSES not NONE
    with pytest.raises(AssertionError):
        XMLDatasetSubClass(**dataconfig)
