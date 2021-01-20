import copy
import warnings


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        if cfg['type'].startswith('Load'):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg
