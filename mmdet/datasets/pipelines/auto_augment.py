import copy

import numpy as np

from mmdet.core import BitmapMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in
    "Learning Data Augmentation Strategies for Object Detection".
    Link: https://arxiv.org/pdf/1906.11172.pdf
    Required key is "img", and optional keys are "gt_bboxes", "gt_masks",
    "gt_seg".

    Args:
        auto_augment_policies (list[list[dict]]): The policies of auto
            augmentation. Each element in auto_augment_policies is a
            specifically augmentation policy, and is composed by several
            augmentations (dict). When call AutoAugment, a random element in
            auto_augment_policies will be selected to augment the image.
            Defaults: None.

    Examples:
        >>> replace = (104, 116, 124)
        >>> auto_augment_policies = [
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
        >>> augmentation = AutoAugment(auto_augment_policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, auto_augment_policies=None):
        assert type(auto_augment_policies) is list and len(
            auto_augment_policies) > 0, \
            'The type and length of auto_augment_policies' \
            'must be list and more than 0, respectively.'
        for one_augment_policy in auto_augment_policies:
            assert type(one_augment_policy) is list and len(
                one_augment_policy) > 0, \
                'The type and length of each element in' \
                'auto_augment_policies must be list and more than 0, ' \
                'respectively.'
            for augment in one_augment_policy:
                assert type(augment) is dict and 'type' in augment, \
                    'The type of each specifically augmentation must be dict' \
                    "and contain the key of 'type'."

        self.auto_augment_policies = auto_augment_policies.copy()
        raise NotImplementedError('Auto augmentation is working in progress '
                                  'and currently not callable.')

    def __call__(self, results):
        if self.auto_augment_policies is not None:
            img = results['img']
            gt_bboxes = results.get('gt_bboxes', None)
            gt_masks = results.get('gt_masks', None)
            gt_seg = results.get('gt_seg', None)

            img, gt_bboxes, gt_masks, gt_seg = self.auto_augment_core(
                img, gt_bboxes, gt_masks, gt_seg, self.auto_augment_policies)

            results['img'] = img
            if 'gt_bboxes' in results:
                results['gt_bboxes'] = gt_bboxes
            if 'gt_masks' in results:
                results['gt_masks'] = gt_masks
            if 'gt_seg' in results:
                results['gt_seg'] = gt_seg

        return results

    def auto_augment_core(self, img, gt_bboxes, gt_masks, gt_seg, policies):
        h, w, c = img.shape
        hw_rescale = np.array([[h, w, h, w]])
        # from x1y1x2y2 order to y1x1y2x2 order
        gt_bboxes = swap_box(gt_bboxes)
        # normalize bboxes
        gt_bboxes /= hw_rescale
        gt_masks = copy.deepcopy(gt_masks)
        gt_masks = getattr(gt_masks, 'masks', None)
        gt_seg = copy.deepcopy(gt_seg)

        # auto augment
        select_policies = policies[np.random.randint(len(policies))]
        for policy in select_policies:
            policy = policy.copy()
            p = eval(policy.pop('type'))(**policy)
            img, gt_bboxes, gt_masks, gt_seg = p(img, gt_bboxes, gt_masks,
                                                 gt_seg)

        h, w, c = img.shape
        wh_rescale = np.array([[w, h, w, h]])
        # from y1x1y2x2 order to x1y1x2y2 order
        gt_bboxes = swap_box(gt_bboxes)
        gt_bboxes *= wh_rescale
        if gt_masks is not None:
            gt_masks = BitmapMasks(gt_masks, h, w)

        return img, gt_bboxes, gt_masks, gt_seg

    def __repr__(self):
        return f'{self.__class__.__name__}' \
            f'(auto_augment_policies={self.auto_augment_policies}'


def swap_box(bboxes):
    '''Swap bounding bboxes from x1y1x2y2(y1x1y2x2) order to
    y1x1y2x2(x1y1x2y2) order.

    Args:
        bboxes (np.array): The bounding boxes with shape (N, 4).

    Returns:
        np.array: The swapped bounding boxes
    '''
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0::2] = bboxes[:, 1::2]
    new_bboxes[:, 1::2] = bboxes[:, 0::2]
    return new_bboxes
