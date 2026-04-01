# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from unittest import TestCase

from mmdet.datasets.utils import validate_pipeline_order


class TestValidatePipelineOrder(TestCase):
    """Tests for :func:`validate_pipeline_order`."""

    def test_correct_order_no_warning(self):
        """A standard pipeline in the correct order should not warn."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
            dict(type='PackDetInputs'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            assert len(w) == 0, [str(x.message) for x in w]

    def test_normalize_before_random_flip_warns(self):
        """Normalize before RandomFlip should trigger a warning."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800)),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            msgs = [str(x.message) for x in w]
            assert any('RandomFlip' in m and 'Normalize' in m for m in msgs)

    def test_pad_before_resize_warns(self):
        """Pad appearing before Resize is suspicious."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Pad', size_divisor=32),
            dict(type='Resize', scale=(1333, 800)),
            dict(type='PackDetInputs'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            msgs = [str(x.message) for x in w]
            assert any('Pad' in m and 'Resize' in m for m in msgs)

    def test_empty_pipeline_no_error(self):
        """An empty pipeline should not crash."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order([])
            assert len(w) == 0

    def test_minimal_pipeline_no_warning(self):
        """A minimal valid pipeline should not warn."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            assert len(w) == 0, [str(x.message) for x in w]

    def test_unknown_transforms_ignored(self):
        """Transforms not in the known list should be silently skipped."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='MyCustomTransform'),
            dict(type='Resize', scale=(1333, 800)),
            dict(type='PackDetInputs'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            assert len(w) == 0, [str(x.message) for x in w]

    def test_issue_6106_bad_pipeline_warns(self):
        """The problematic pipeline from issue #6106 should trigger
        warnings. The user had Normalize before Pad."""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=[(800, 200), (800, 700)]),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate_pipeline_order(pipeline)
            # Should warn about Pad after Normalize (or Normalize before Pad)
            assert len(w) > 0, 'Expected warnings for issue #6106 pipeline'
            msgs = [str(x.message) for x in w]
            assert any('Pad' in m and 'Normalize' in m for m in msgs)
