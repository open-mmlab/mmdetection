import os.path as osp
import tempfile
import unittest

from mmengine.fileio import dump

from mmdet.datasets.api_wrappers import COCOPanoptic


class TestCOCOPanoptic(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_create_index(self):
        ann_json = {'test': ['test', 'createIndex']}
        annotation_file = osp.join(self.tmp_dir.name, 'createIndex.json')
        dump(ann_json, annotation_file)
        COCOPanoptic(annotation_file)

    def test_load_anns(self):
        categories = [{
            'id': 0,
            'name': 'person',
            'supercategory': 'person',
            'isthing': 1
        }]

        images = [{
            'id': 0,
            'width': 80,
            'height': 60,
            'file_name': 'fake_name1.jpg',
        }]

        annotations = [{
            'segments_info': [
                {
                    'id': 1,
                    'category_id': 0,
                    'area': 400,
                    'bbox': [10, 10, 10, 40],
                    'iscrowd': 0
                },
            ],
            'file_name':
            'fake_name1.png',
            'image_id':
            0
        }]

        ann_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

        annotation_file = osp.join(self.tmp_dir.name, 'load_anns.json')
        dump(ann_json, annotation_file)

        api = COCOPanoptic(annotation_file)
        api.load_anns(1)

        self.assertIsNone(api.load_anns(0.1))
