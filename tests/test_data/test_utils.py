import pytest

from mmdet.datasets import replace_ImageToTensor


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
