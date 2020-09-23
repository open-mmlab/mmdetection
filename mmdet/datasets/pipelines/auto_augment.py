import copy

import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    TODO: Implement 'Shear', 'Sharpness' and 'Rotate' transforms

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
        return f'{self.__class__.__name__}(policies={self.policies})'


@PIPELINES.register_module()
class Shear(object):
    """Apply Shear Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range [0,_MAX_LEVEL].
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for performing Shear and should be in
            range [0, 1].
        direction (str): The direction for shear, either "horizontal"
            or "vertical".
        max_shear_magnitude (float): The maximum magnitude for Shear
            transformation.
        random_negative_prob (float): The probability that turns the
                offset negative. Should be in range [0,1]
        interpolation (str): Same as in :func:`mmcv.imshear`.
    """

    def __init__(self,
                 level,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 direction='horizontal',
                 max_shear_magnitude=0.3,
                 random_negative_prob=0.5,
                 interpolation='bilinear'):
        assert isinstance(level, (int, float)), 'The level must be type ' \
            f'int or float, got {type(level)}.'
        assert 0 <= level <= _MAX_LEVEL, 'The level should be in range ' \
            f'[0,{_MAX_LEVEL}], got {level}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must ' \
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), 'all ' \
            'elements of img_fill_val should between range [0,255].' \
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability of shear should be in ' \
            f'range [0,1]. got {prob}.'
        assert direction in ('horizontal', 'vertical'), 'direction must ' \
            f'in be either "horizontal" or "vertical". got {direction}.'
        assert isinstance(max_shear_magnitude, float), 'max_shear_magnitude ' \
            f'should be type float. got {type(max_shear_magnitude)}.'
        assert 0. <= max_shear_magnitude <= 1., 'Defaultly ' \
            'max_shear_magnitude should be in range [0,1]. ' \
            f'got {max_shear_magnitude}.'
        self.level = level
        self.magnitude = level_to_value(level, max_shear_magnitude)
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.direction = direction
        self.max_shear_magnitude = max_shear_magnitude
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def _shear_img(self,
                   results,
                   magnitude,
                   direction='horizontal',
                   interpolation='bilinear'):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation)
            results[key] = img_sheared.astype(img.dtype)

    def _shear_bboxes(self, results, magnitude):
        """Shear the bboxes."""
        h, w, c = results['img_shape']
        if self.direction == 'horizontal':
            shear_matrix = np.stack([[1, magnitude],
                                     [0, 1]]).astype(np.float32)  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude,
                                              1]]).astype(np.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose(
                (2, 1, 0)).astype(np.float32)  # [nb_box, 2, 4]
            new_coords = np.matmul(shear_matrix[None, :, :],
                                   coordinates)  # [nb_box, 2, 4]
            min_x = np.min(new_coords[:, 0, :], axis=-1)
            min_y = np.min(new_coords[:, 1, :], axis=-1)
            max_x = np.max(new_coords[:, 0, :], axis=-1)
            max_y = np.max(new_coords[:, 1, :], axis=-1)
            min_x = np.clip(min_x, a_min=0, a_max=w)
            min_y = np.clip(min_y, a_min=0, a_max=h)
            max_x = np.clip(max_x, a_min=min_x, a_max=w)
            max_y = np.clip(max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _shear_masks(self,
                     results,
                     magnitude,
                     direction='horizontal',
                     fill_val=0,
                     interpolation='bilinear'):
        """Shear the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction,
                                       border_value=fill_val,
                                       interpolation=interpolation)

    def _shear_seg(self,
                   results,
                   magnitude,
                   direction='horizontal',
                   fill_val=255,
                   interpolation='bilinear'):
        """Shear the segmentation maps."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            results[key] = mmcv.imshear(
                seg,
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after shear
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to shear images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Sheared results.
        """
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        self._shear_img(results, magnitude, self.direction, self.interpolation)
        self._shear_bboxes(results, magnitude)
        # fill_val set to 0 for background of mask.
        self._shear_masks(
            results,
            magnitude,
            self.direction,
            fill_val=0,
            interpolation=self.interpolation)
        self._shear_seg(
            results,
            magnitude,
            self.direction,
            fill_val=self.seg_ignore_label,
            interpolation=self.interpolation)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'max_shear_magnitude={self.max_shear_magnitude}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
