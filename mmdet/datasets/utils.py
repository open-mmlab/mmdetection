# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.transforms import LoadImageFromFile

from mmdet.datasets.transforms import LoadAnnotations, LoadPanopticAnnotations
from mmdet.registry import TRANSFORMS

# Categories used by validate_pipeline_order to detect suspicious ordering.
# Each transform is assigned to one category; the expected order is the
# numeric order of those categories.  Placing a higher-category transform
# before a lower-category one triggers a warning.
_TRANSFORM_CATEGORIES = {
    # --- Loading (must come first) ---
    'LoadImageFromFile': 0,
    'LoadMultiChannelImageFromFiles': 0,
    'LoadAnnotations': 1,
    'LoadPanopticAnnotations': 1,
    'LoadProposals': 1,
    # --- Spatial augmentation ---
    'Resize': 10,
    'FixScaleResize': 10,
    'FixShapeResize': 10,
    'ResizeShortestEdge': 10,
    'RandomResize': 10,
    'RandomCrop': 11,
    'RandomCenterCropPad': 11,
    'MinIoURandomCrop': 11,
    'Expand': 11,
    'RandomFlip': 12,
    'RandomShift': 12,
    'RandomAffine': 12,
    'Mosaic': 13,
    'CachedMosaic': 13,
    'MixUp': 13,
    'CachedMixUp': 13,
    'CopyPaste': 13,
    # --- Pad ---
    'Pad': 20,
    # --- Pixel-level augmentation (after spatial) ---
    'PhotoMetricDistortion': 15,
    'YOLOXHSVRandomAug': 15,
    'Albu': 15,
    'InstaBoost': 15,
    # --- Normalize (after all augmentation) ---
    'Normalize': 30,
    # --- Formatting (must come last) ---
    'DefaultFormatBundle': 40,
    'PackDetInputs': 40,
    'PackTrackInputs': 40,
    'PackReIDInputs': 40,
    'Collect': 41,
}

# Specific pair-wise rules: (earlier, later) means ``earlier`` must come
# before ``later`` in the pipeline.  Each entry carries a human-readable
# reason shown in the warning message.
_ORDER_RULES = [
    ('LoadImageFromFile', 'LoadAnnotations',
     'Annotations should be loaded after the image.'),
    ('LoadImageFromFile', 'Resize',
     'Image must be loaded before resizing.'),
    ('Resize', 'Pad',
     'Padding should happen after resizing; otherwise the resize may '
     'undo the padding.'),
    ('Resize', 'Normalize',
     'Normalize should be applied after spatial transforms like Resize.'),
    ('RandomFlip', 'Normalize',
     'Normalize should come after spatial augmentations like RandomFlip.'),
    ('RandomFlip', 'Pad',
     'Pad should come after RandomFlip to avoid padding artifacts being '
     'flipped.'),
    ('Normalize', 'PackDetInputs',
     'Formatting / packing must be the last step.'),
    ('Normalize', 'DefaultFormatBundle',
     'Formatting / packing must be the last step.'),
    ('Pad', 'Normalize',
     'Normalize should be applied after Pad so that padded values are '
     'also normalized.  If you intentionally normalize before padding, '
     'you can ignore this warning.'),
]


def validate_pipeline_order(pipeline):
    """Check that the transforms in *pipeline* follow a sensible order.

    The function emits :py:class:`UserWarning` messages for each suspicious
    ordering it detects.  It never raises an exception so that existing
    configs keep working.

    Args:
        pipeline (list[dict]): Data pipeline config list, where each
            element is a ``dict(type=..., ...)``.

    Example:
        >>> pipeline = [
        ...     dict(type='LoadImageFromFile'),
        ...     dict(type='Normalize', mean=[0]*3, std=[1]*3),
        ...     dict(type='RandomFlip', prob=0.5),  # after Normalize!
        ...     dict(type='PackDetInputs'),
        ... ]
        >>> validate_pipeline_order(pipeline)  # emits a warning
    """
    if not pipeline:
        return

    # Build a mapping from transform name to its *first* index in the
    # pipeline so that we can check pair-wise rules.
    first_index = {}
    for idx, step in enumerate(pipeline):
        name = step.get('type', '')
        if name and name not in first_index:
            first_index[name] = idx

    # --- pair-wise rule checks ---
    for early, late, reason in _ORDER_RULES:
        early_idx = first_index.get(early)
        late_idx = first_index.get(late)
        if early_idx is not None and late_idx is not None:
            if early_idx > late_idx:
                warnings.warn(
                    f"In the data pipeline, '{early}' (index {early_idx}) "
                    f"appears after '{late}' (index {late_idx}), which is "
                    f"likely incorrect. {reason}  Please review the "
                    f"transform order in your config.",
                    UserWarning,
                )

    # --- generic category-order check ---
    prev_name = None
    prev_cat = -1
    for step in pipeline:
        name = step.get('type', '')
        cat = _TRANSFORM_CATEGORIES.get(name)
        if cat is None:
            # Unknown transform – skip.
            continue
        if cat < prev_cat:
            warnings.warn(
                f"In the data pipeline, '{name}' (category {cat}) appears "
                f"after '{prev_name}' (category {prev_cat}).  The "
                f"recommended order is: Loading -> Spatial augmentation "
                f"-> Pixel augmentation -> Pad -> Normalize -> "
                f"Formatting.  Please verify the transform order in "
                f"your config.",
                UserWarning,
            )
        prev_name = name
        prev_cat = cat


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
        obj_cls = TRANSFORMS.get(cfg['type'])
        # TODO:use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg
