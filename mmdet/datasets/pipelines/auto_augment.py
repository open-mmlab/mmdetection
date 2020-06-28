import copy

import numpy as np
import pycocotools.mask as mask_util

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in
    `Learning Data Augmentation Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_  # noqa: E501
    Required key is "img", and optional keys are "gt_bboxes", "gt_masks",
    "gt_seg".

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

    def __init__(self, policies=None):
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
        if self.policies is not None:
            img = results['img']
            gt_bboxes = results.get('gt_bboxes', None)
            gt_masks = results.get('gt_masks', None)
            gt_seg = results.get('gt_seg', None)

            img, gt_bboxes, gt_masks, gt_seg = self.auto_augment_core(
                img, gt_bboxes, gt_masks, gt_seg, self.policies)

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
        wh_rescale = np.array([[w, h, w, h]])
        # normalize bboxes
        gt_bboxes /= wh_rescale
        gt_masks = gt_masks.copy()
        if gt_masks is not None:
            gt_masks_numpy = gt_masks.to_ndarray()
        gt_seg = gt_seg.copy()

        # auto augment
        select_policies = policies[np.random.randint(len(policies))]
        for policy in select_policies:
            policy = policy.copy()
            p = eval(policy.pop('type'))(**policy)
            img, gt_bboxes, gt_masks_numpy, gt_seg = p(img, gt_bboxes,
                                                       gt_masks_numpy, gt_seg)

        h, w, c = img.shape
        wh_rescale = np.array([[w, h, w, h]])
        gt_bboxes *= wh_rescale
        if isinstance(gt_masks, BitmapMasks):
            gt_masks = BitmapMasks(gt_masks_numpy, h, w)
        elif isinstance(gt_masks, PolygonMasks):
            encoded_gt_masks = []
            for i in range(len(gt_masks)):
                encoded_gt_masks.append(
                    mask_util.encode(
                        np.array(
                            gt_masks_numpy[i, :, :, np.newaxis],
                            order='F',
                            dtype='uint8'))[0])  # encoded with RLE
            gt_masks = PolygonMasks(encoded_gt_masks, h, w)
        return img, gt_bboxes, gt_masks, gt_seg

    def __repr__(self):
        return f'{self.__class__.__name__}' \
            f'(policies={self.policies}'
