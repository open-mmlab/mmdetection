import copy

import cv2
import numpy as np
import pycocotools.mask as mask_util

from mmdet.core.mask import BitmapMasks, PolygonMasks
from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in
    `Learning Data Augmentation Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_  # noqa: E501

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.

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
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies}'


@PIPELINES.register_module()
class Translate(object):
    """Translate images along with x-axis or y-axis.

    Args:
        base_offset (float): base_offset * 25.0 is the actual offset of
            translation.
        prob (float): The probability for perform translating and should be in
            range 0 to 1.
        replace (tuple): The filled values of image border area and should
            have 3 elements.
        axis (str): Translate images along with x-axis or y-axis. The option
            of axis is 'x' or 'y'.
    """

    def __init__(self, base_offset, prob, replace, axis):
        assert prob >= 0 and prob <= 1.0, \
            'The probability of translation should be in range 0 to 1.'
        assert isinstance(replace, tuple) and len(replace) == 3, \
            'replace must be a tuple with length 3.'
        assert axis in ('x', 'y'), \
            'Translate should be alone with x-axis or y-axis.'
        self.base_offset = base_offset
        self.prob = prob
        self.replace = replace
        self.axis = axis

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        # the offset of translation
        offset = self.base_offset * 25.0
        offset = -offset if np.random.rand() < 0.5 else offset
        # the transformation matrix of cv2
        if self.axis == 'x':
            trans_matrix = np.float32([[1, 0, offset], [0, 1, offset]])
        else:
            trans_matrix = np.float32([[1, 0, 0], [0, 1, offset]])

        self._translate_img(results, trans_matrix, replace=self.replace)
        self._translate_bboxes(results, offset)
        self._translate_masks(results, trans_matrix, replace=0)
        self._translate_seg(results, trans_matrix, replace=255)
        return results

    def _translate_img(self, results, trans_matrix, replace):
        for key in results.get('img_fields', ['img']):
            results[key] = cv2.warpAffine(
                results[key],
                trans_matrix,
                results[key].size,
                borderValue=replace)

    def _translate_bboxes(self, results, offset):
        H, W, C = results['img_shape']
        for key in results.get('box_fields', []):
            min_x, min_y, max_x, max_y = results[key]
            if self.axis == 'x':
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(W, max_x + offset)
            else:
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(H, max_y + offset)

            # clip box
            min_x = np.clip(min_x, 0, W)
            min_y = np.clip(min_y, 0, H)
            max_x = np.clip(max_x, 0, W)
            max_y = np.clip(max_y, 0, H)
            # boxes should have min area
            h = max_y - min_y
            w = max_x - min_x
            min_x[w == 0] = np.minimum(min_x[w == 0], (1 - 0.05) * W)
            max_x[w == 0] = np.maximum(max_x[w == 0], (0 + 0.05) * W)
            min_y[h == 0] = np.minimum(min_y[h == 0], (1 - 0.05) * H)
            max_y[h == 0] = np.maximum(max_y[h == 0], (0 + 0.05) * H)

            results[key] = np.stack([min_x, min_y, max_x, max_y], -1)

    def _translate_mask(self, results, trans_matrix, replace=0):
        H, W, C = results['img_shape']
        for key in results.get('mask_fields', []):
            translate_masks = []
            for mask in results[key].to_ndarray():
                translate_mask = cv2.warpAffine(
                    mask,
                    trans_matrix,
                    mask.size,
                    flags=cv2.INTER_NEAREST,
                    borderValue=replace)

                if isinstance(results[key], BitmapMasks):
                    translate_masks.append(translate_mask)
                elif isinstance(results[key], PolygonMasks):
                    # encoded with RLE
                    translate_masks.append(
                        mask_util.encode(
                            np.array(
                                translate_mask[:, :, np.newaxis],
                                order='F',
                                dtype='uint8'))[0])

            if isinstance(results[key], BitmapMasks):
                results[key] = BitmapMasks(
                    np.concatenate(translate_masks), H, W)
            elif isinstance(results[key], PolygonMasks):
                results[key] = PolygonMasks(translate_masks, H, W)

    def _translate_seg(self, results, trans_matrix, replace=255):
        for key in results.get('seg_fields', []):
            results[key] = cv2.warpAffine(
                results[key],
                trans_matrix,
                results[key].size,
                flags=cv2.INTER_NEAREST,
                borderValue=replace)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(base_offset={self.base_offset}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'replace={self.replace}, '
        repr_str += f'axis={self.axis})'
        return repr_str
