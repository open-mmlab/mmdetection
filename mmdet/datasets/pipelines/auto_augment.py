import copy

import numpy as np

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in
    `Learning Data Augmentation Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_  # noqa: E501
    Required key is "img", and optional keys are "gt_bboxes", "gt_masks",
    "gt_semantic_seg".

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            element in policies is a specifically augmentation policy, and is
            composed by several augmentations (dict). When call AutoAugment, a
            random element in policies will be selected to augment the image.
            Defaults: None.

    Examples:
        >>> replace = (104, 116, 124)
        >>> policies = [
        >>>     [
        >>>         dict(type='Sharpness', prob=0.0, level=8),
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0,
        >>>             replace=replace,
        >>>             axis='x')
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10,
        >>>             replace=replace),
        >>>         dict(type='Color', prob=1.0, level=6)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, policies):
        assert isinstance(policies, list) and len(policies) > 0, \
            'The type and length of policies must be list and more than 0, ' \
            'respectively.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'The type and length of each element in policies must be' \
                'list and more than 0, respectively.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'The type of each specifically augmentation must be' \
                    'dict and contain the key of "type".'

        self.policies = copy.deepcopy(policies)

    def __call__(self, results):
        select_policies = self.policies[np.random.randint(len(self.policies))]
        augmentation = Compose(select_policies)
        return augmentation(results)

    def __repr__(self):
        return f'{self.__class__.__name__}' \
            f'(policies={self.policies}'
