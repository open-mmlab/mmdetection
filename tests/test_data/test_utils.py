import pytest

from mmdet.datasets import get_loading_pipeline, replace_ImageToTensor


def test_replace_ImageToTensor():
    # with MultiScaleFlipAug
    pipelines = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize'),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    expected_pipelines = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize'),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img']),
            ])
    ]
    with pytest.warns(UserWarning):
        assert expected_pipelines == replace_ImageToTensor(pipelines)

    # without MultiScaleFlipAug
    pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize'),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ]
    expected_pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize'),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img']),
    ]
    with pytest.warns(UserWarning):
        assert expected_pipelines == replace_ImageToTensor(pipelines)


def test_get_loading_pipeline():
    pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    expected_pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True)
    ]
    assert expected_pipelines == \
           get_loading_pipeline(pipelines)
