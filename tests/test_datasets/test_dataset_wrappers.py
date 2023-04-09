import unittest

from mmdet.datasets import CocoDataset, MultiImageMixDataset


class TestDatasetWrapper(unittest.TestCase):

    def test_skip_type_keys(self):
        # test CocoDataset
        metainfo = dict(classes=('bus', 'car'), task_name='new_task')
        dataset = CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        multi_dataset = MultiImageMixDataset(
            dataset=dataset,
            pipeline=[],
        )

        skip_type_keys = ('Mosaic', 'RandomAffine', 'MixUp')
        self.assertFalse(multi_dataset.has_all_skip_type_keys(skip_type_keys))

        multi_dataset.update_skip_type_keys(skip_type_keys)
        self.assertTrue(multi_dataset.has_all_skip_type_keys(skip_type_keys))
