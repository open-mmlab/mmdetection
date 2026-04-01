# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest.mock import MagicMock

from mmdet.datasets.dataset_wrappers import ConcatDataset


def _make_mock_dataset(classes, palette=None):
    """Create a mock dataset with the given classes and palette."""
    ds = MagicMock()
    meta = {'classes': tuple(classes)}
    if palette is not None:
        meta['palette'] = list(palette)
    ds.metainfo = meta
    ds.__len__ = MagicMock(return_value=5)
    return ds


class TestConcatDataset(unittest.TestCase):

    def test_merge_metainfo_same_classes(self):
        """Datasets with identical classes should keep a single metainfo."""
        ds1 = _make_mock_dataset(['cat', 'dog'])
        ds2 = _make_mock_dataset(['cat', 'dog'])
        concat = ConcatDataset.__new__(ConcatDataset)
        concat.datasets = [ds1, ds2]
        concat.ignore_keys = []
        merged = concat._merge_metainfo()
        # Same classes: merged dict should equal first dataset's metainfo
        self.assertIsInstance(merged, dict)
        self.assertEqual(merged['classes'], ('cat', 'dog'))

    def test_merge_metainfo_different_classes(self):
        """Datasets with different classes should produce a merged set."""
        ds1 = _make_mock_dataset(
            ['cat', 'dog'],
            palette=[(255, 0, 0), (0, 255, 0)])
        ds2 = _make_mock_dataset(
            ['dog', 'bird'],
            palette=[(0, 255, 0), (0, 0, 255)])
        concat = ConcatDataset.__new__(ConcatDataset)
        concat.datasets = [ds1, ds2]
        concat.ignore_keys = []
        merged = concat._merge_metainfo()
        self.assertIsInstance(merged, dict)
        # 'cat' from ds1, 'dog' shared, 'bird' from ds2
        self.assertEqual(merged['classes'], ('cat', 'dog', 'bird'))
        # palette should carry over from each dataset
        self.assertEqual(merged['palette'][0], (255, 0, 0))  # cat from ds1
        self.assertEqual(merged['palette'][1], (0, 255, 0))  # dog from ds1
        self.assertEqual(merged['palette'][2], (0, 0, 255))  # bird from ds2

    def test_merge_metainfo_no_overlap(self):
        """Datasets with completely disjoint classes."""
        ds1 = _make_mock_dataset(['a', 'b'])
        ds2 = _make_mock_dataset(['c', 'd'])
        concat = ConcatDataset.__new__(ConcatDataset)
        concat.datasets = [ds1, ds2]
        concat.ignore_keys = []
        merged = concat._merge_metainfo()
        self.assertIsInstance(merged, dict)
        self.assertEqual(merged['classes'], ('a', 'b', 'c', 'd'))

    def test_merge_metainfo_missing_classes_key(self):
        """If a dataset has no 'classes' key, fall back to list of dicts."""
        ds1 = _make_mock_dataset(['cat'])
        ds2 = MagicMock()
        ds2.metainfo = {'custom_key': 'value'}
        concat = ConcatDataset.__new__(ConcatDataset)
        concat.datasets = [ds1, ds2]
        concat.ignore_keys = []
        merged = concat._merge_metainfo()
        # Should fall back to list
        self.assertIsInstance(merged, list)
        self.assertEqual(len(merged), 2)


if __name__ == '__main__':
    unittest.main()
