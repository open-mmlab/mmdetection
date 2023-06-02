# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
from mmcv.transforms import RandomChoice
from mmcv.transforms.utils import cache_randomness
from mmengine.config import ConfigDict

from mmdet.registry import TRANSFORMS

# AutoAugment uses reinforcement learning to search for
# some widely useful data augmentation strategies,
# here we provide AUTOAUG_POLICIES_V0.
# For AUTOAUG_POLICIES_V0, each tuple is an augmentation
# operation of the form (operation, probability, magnitude).
# Each element in policies is a policy that will be applied
# sequentially on the image.

# RandAugment defines a data augmentation search space, RANDAUG_SPACE,
# sampling 1~3 data augmentations each time, and
# setting the magnitude of each data augmentation randomly,
# which will be applied sequentially on the image.

_MAX_LEVEL = 10

AUTOAUG_POLICIES_V0 = [
    [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
    [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
    [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
    [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
    [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
    [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
    [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
    [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
    [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
    [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
    [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
    [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
    [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
    [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
    [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
    [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
    [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
    [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
    [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
    [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
    [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
    [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
    [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
    [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
    [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
]


def policies_v0():
    """Autoaugment policies that was used in AutoAugment Paper."""
    policies = list()
    for policy_args in AUTOAUG_POLICIES_V0:
        policy = list()
        for args in policy_args:
            policy.append(dict(type=args[0], prob=args[1], level=args[2]))
        policies.append(policy)
    return policies


RANDAUG_SPACE = [[dict(type='AutoContrast')], [dict(type='Equalize')],
                 [dict(type='Invert')], [dict(type='Rotate')],
                 [dict(type='Posterize')], [dict(type='Solarize')],
                 [dict(type='SolarizeAdd')], [dict(type='Color')],
                 [dict(type='Contrast')], [dict(type='Brightness')],
                 [dict(type='Sharpness')], [dict(type='ShearX')],
                 [dict(type='ShearY')], [dict(type='TranslateX')],
                 [dict(type='TranslateY')]]


def level_to_mag(level: Optional[int], min_mag: float,
                 max_mag: float) -> float:
    """Map from level to magnitude."""
    if level is None:
        return round(np.random.rand() * (max_mag - min_mag) + min_mag, 1)
    else:
        return round(level / _MAX_LEVEL * (max_mag - min_mag) + min_mag, 1)


@TRANSFORMS.register_module()
class AutoAugment(RandomChoice):
    """Auto augmentation.

    This data augmentation is proposed in `AutoAugment: Learning
    Augmentation Policies from Data <https://arxiv.org/abs/1805.09501>`_
    and in `Learning Data Augmentation Strategies for Object Detection
    <https://arxiv.org/pdf/1906.11172>`_.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_bboxes_labels
    - gt_masks
    - gt_ignore_flags
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        policies (List[List[Union[dict, ConfigDict]]]):
            The policies of auto augmentation.Each policy in ``policies``
            is a specific augmentation policy, and is composed by several
            augmentations. When AutoAugment is called, a random policy in
            ``policies`` will be selected to augment images.
            Defaults to policy_v0().
        prob (list[float], optional): The probabilities associated
            with each policy. The length should be equal to the policy
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed. Defaults to None.

    Examples:
        >>> policies = [
        >>>     [
        >>>         dict(type='Sharpness', prob=0.0, level=8),
        >>>         dict(type='ShearX', prob=0.4, level=0,)
        >>>     ],
        >>>     [
        >>>         dict(type='Rotate', prob=0.6, level=10),
        >>>         dict(type='Color', prob=1.0, level=6)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self,
                 policies: List[List[Union[dict, ConfigDict]]] = policies_v0(),
                 prob: Optional[List[float]] = None) -> None:
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'
        super().__init__(transforms=policies, prob=prob)
        self.policies = policies

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(policies={self.policies}, ' \
               f'prob={self.prob})'


@TRANSFORMS.register_module()
class RandAugment(RandomChoice):
    """Rand augmentation.

    This data augmentation is proposed in `RandAugment:
    Practical automated data augmentation with a reduced
    search space <https://arxiv.org/abs/1909.13719>`_.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_bboxes_labels
    - gt_masks
    - gt_ignore_flags
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        aug_space (List[List[Union[dict, ConfigDict]]]): The augmentation space
            of rand augmentation. Each augmentation transform in ``aug_space``
            is a specific transform, and is composed by several augmentations.
            When RandAugment is called, a random transform in ``aug_space``
            will be selected to augment images. Defaults to aug_space.
        aug_num (int): Number of augmentation to apply equentially.
            Defaults to 2.
        prob (list[float], optional): The probabilities associated with
            each augmentation. The length should be equal to the
            augmentation space and the sum should be 1. If not given,
            a uniform distribution will be assumed. Defaults to None.

    Examples:
        >>> aug_space = [
        >>>     dict(type='Sharpness'),
        >>>     dict(type='ShearX'),
        >>>     dict(type='Color'),
        >>>     ],
        >>> augmentation = RandAugment(aug_space)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self,
                 aug_space: List[Union[dict, ConfigDict]] = RANDAUG_SPACE,
                 aug_num: int = 2,
                 prob: Optional[List[float]] = None) -> None:
        assert isinstance(aug_space, list) and len(aug_space) > 0, \
            'Augmentation space must be a non-empty list.'
        for aug in aug_space:
            assert isinstance(aug, list) and len(aug) == 1, \
                'Each augmentation in aug_space must be a list.'
            for transform in aug:
                assert isinstance(transform, dict) and 'type' in transform, \
                    'Each specific transform must be a dict with key' \
                    ' "type".'
        super().__init__(transforms=aug_space, prob=prob)
        self.aug_space = aug_space
        self.aug_num = aug_num

    @cache_randomness
    def random_pipeline_index(self):
        indices = np.arange(len(self.transforms))
        return np.random.choice(
            indices, self.aug_num, p=self.prob, replace=False)

    def transform(self, results: dict) -> dict:
        """Transform function to use RandAugment.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with RandAugment.
        """
        for idx in self.random_pipeline_index():
            results = self.transforms[idx](results)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'aug_space={self.aug_space}, '\
               f'aug_num={self.aug_num}, ' \
               f'prob={self.prob})'
