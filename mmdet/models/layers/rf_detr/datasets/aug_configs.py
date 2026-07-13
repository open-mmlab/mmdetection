# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Augmentation presets and default configuration for RF-DETR training.

Import a preset and pass it as ``aug_config`` to your training call:

```python
from mmdet.models.layers.rf_detr.datasets.aug_configs import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="...", aug_config=AUG_CONSERVATIVE) model.train(dataset_dir="...", aug_config=AUG_AGGRESSIVE)

# Disable all augmentations
model.train(dataset_dir="...", aug_config={})

# Fully custom
model.train(dataset_dir="...", aug_config={"HorizontalFlip": {"p": 0.5}})
```

## Available presets

| Preset         | Best for                                         |
| -------------- | ------------------------------------------------ |
| ``AUG_CONSERVATIVE``  | Small datasets (under 500 images)             |
| ``AUG_AGGRESSIVE``    | Large datasets (2000+ images)                 |
| ``AUG_AERIAL``        | Satellite / overhead imagery                  |
| ``AUG_INDUSTRIAL``    | Manufacturing / inspection data               |

## Transform Categories

**Geometric transforms** (automatically transform bounding boxes):
- Flips: HorizontalFlip, VerticalFlip
- Rotations: Rotate, Affine, ShiftScaleRotate
- Crops: RandomCrop, CenterCrop, RandomResizedCrop
- Perspective: Perspective, ElasticTransform, GridDistortion

**Pixel-level transforms** (preserve bounding boxes):
- Color: ColorJitter, HueSaturationValue, RandomBrightnessContrast
- Blur/Noise: GaussianBlur, GaussNoise, Blur
- Enhancement: CLAHE, Sharpen, Equalize

## Best Practices

1. **Start conservative**: Use moderate probabilities (p=0.3-0.5) and small parameter ranges
2. **Geometric caution**: Extreme rotations (>45°) or crops may remove too many boxes
3. **Performance**: Fewer transforms = faster training; prioritize transforms that match your domain
4. **Validation**: Monitor validation mAP - excessive augmentation can hurt performance
5. **Domain-specific**: Enable augmentations that reflect real-world variations in your data

## Adding Custom Transforms

For geometric transforms not in GEOMETRIC_TRANSFORMS set, add them in transforms.py:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",  # Add here
}
```

## Kornia GPU Backend

When ``augmentation_backend="auto"`` or ``"gpu"`` is set in ``TrainConfig``, augmentations run on the GPU via Kornia
instead of Albumentations.

**Supported transforms** (all presets):

| Preset key | Kornia equivalent | Notes |
|---|---|---|
| ``HorizontalFlip`` | ``K.RandomHorizontalFlip`` | Direct |
| ``VerticalFlip`` | ``K.RandomVerticalFlip`` | Direct |
| ``Rotate`` | ``K.RandomRotation`` | ``limit`` may be scalar or tuple |
| ``Affine`` | ``K.RandomAffine`` | ``translate_percent`` treated as fraction |
| ``ColorJitter`` | ``K.ColorJiggle`` | Same multiplicative semantics |
| ``RandomBrightnessContrast`` | ``K.ColorJiggle`` | ``brightness_limit`` / ``contrast_limit`` direct |
| ``GaussianBlur`` | ``K.RandomGaussianBlur`` | ``blur_limit`` rounded up to odd; ``sigma=(0.1, 2.0)`` |
| ``GaussNoise`` | ``K.RandomGaussianNoise`` | Upper bound of ``std_range`` used as fixed std |

**Phase 1 limitation**: Segmentation models (``segmentation_head=True``) skip GPU augmentation; CPU Albumentations are
used instead. Mask support is planned for Phase 2.
"""

# ---------------------------------------------------------------------------
# Default configuration (backward-compatible baseline)
# ---------------------------------------------------------------------------

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    # "VerticalFlip": {"p": 0.5},
    # "Rotate": {"limit": 15, "p": 0.5},  # Better keep small angles
}

# ---------------------------------------------------------------------------
# Named presets — import and pass directly as aug_config=<preset>
# ---------------------------------------------------------------------------

#: Minimal augmentations — safe for small datasets (under 500 images).
AUG_CONSERVATIVE = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}

#: Aggressive augmentations — for larger datasets (2000+ images).
AUG_AGGRESSIVE = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}

#: Optimised for aerial / satellite imagery (overhead views, 90° rotations).
AUG_AERIAL = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
}

#: Optimised for industrial / manufacturing data (lighting & sensor noise).
AUG_INDUSTRIAL = {
    "HorizontalFlip": {"p": 0.3},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},
}
