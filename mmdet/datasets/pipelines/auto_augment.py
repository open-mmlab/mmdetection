import copy

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


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


@PIPELINES.register_module()
class Rotate(object):
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Isotropic scale factor. Same in
            ``mmcv.imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
        random_negative_prob (float): The probability that turns the
             offset negative.
    """

    def __init__(self,
                 level,
                 scale=1,
                 center=None,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 max_rotate_angle=30,
                 random_negative_prob=0.5):
        assert isinstance(level, (int, float)), \
            f'The level must be type int or float. got {type(level)}.'
        assert 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range (0,{_MAX_LEVEL}]. got {level}.'
        assert isinstance(scale, (int, float)), \
            f'The scale must be type int or float. got type {type(scale)}.'
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            'got {prob}.'
        assert isinstance(max_rotate_angle, (int, float)), 'max_rotate_angle '\
            f'should be type int or float. got type {type(max_rotate_angle)}.'
        self.level = level
        self.scale = scale
        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle = level_to_value(level, max_rotate_angle)
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle
        self.random_negative_prob = random_negative_prob

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val)
            results[key] = img_rotated.astype(img.dtype)

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (coordinates,
                 np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                axis=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose(
                (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix,
                                       coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = np.min(
                rotated_coords[:, :, 0], axis=1), np.min(
                    rotated_coords[:, :, 1], axis=1)
            max_x, max_y = np.max(
                rotated_coords[:, :, 0], axis=1), np.max(
                    rotated_coords[:, :, 1], axis=1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _rotate_masks(self,
                      results,
                      angle,
                      center=None,
                      scale=1.0,
                      fill_val=0):
        """Rotate the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self,
                    results,
                    angle,
                    center=None,
                    scale=1.0,
                    fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale,
                border_value=fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
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
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results['img'].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        angle = random_negative(self.angle, self.random_negative_prob)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_rotate_angle={self.max_rotate_angle}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class Translate(object):
    """Translate the images, bboxes, masks and segmentation maps horizontally
    or vertically.

    Args:
        level (int | float): The level for Translate and should be in
            range [0,_MAX_LEVEL].
        prob (float): The probability for performing translation and
            should be in range [0, 1].
        img_fill_val (int | float | tuple): The filled value for image
            border. If float, the same fill value will be used for all
            the three channels of image. If tuple, the should be 3
            elements (e.g. equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        max_translate_offset (int | float): The maximum pixel's offset for
            Translate.
        random_negative_prob (float): The probability that turns the
            offset negative.
        min_size (int | float): The minimum pixel for filtering
            invalid bboxes after the translation.
    """

    def __init__(self,
                 level,
                 prob=0.5,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 direction='horizontal',
                 max_translate_offset=250.,
                 random_negative_prob=0.5,
                 min_size=0):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range [0,_MAX_LEVEL]'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range [0, 1].'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, \
                'img_fill_val as tuple must have 3 elements.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError('img_fill_val must be type float or tuple.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255].'
        assert direction in ('horizontal', 'vertical'), \
            'direction should be "horizontal" or "vertical".'
        assert isinstance(max_translate_offset, (int, float)), \
            'The max_translate_offset must be type int or float.'
        # the offset used for translation
        self.offset = int(level_to_value(level, max_translate_offset))
        self.level = level
        self.prob = prob
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.direction = direction
        self.max_translate_offset = max_translate_offset
        self.random_negative_prob = random_negative_prob
        self.min_size = min_size

    def _translate_img(self, results, offset, direction='horizontal'):
        """Translate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(
                img, offset, direction, self.img_fill_val).astype(img.dtype)

    def _translate_bboxes(self, results, offset):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            if self.direction == 'horizontal':
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif self.direction == 'vertical':
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y],
                                          axis=-1)

    def _translate_masks(self,
                         results,
                         offset,
                         direction='horizontal',
                         fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self,
                       results,
                       offset,
                       direction='horizontal',
                       fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction,
                                            fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
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
        return results

    def __call__(self, results):
        """Call function to translate images, bounding boxes, masks and
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Translated results.
        """
        if np.random.rand() > self.prob:
            return results
        offset = random_negative(self.offset, self.random_negative_prob)
        self._translate_img(results, offset, self.direction)
        self._translate_bboxes(results, offset)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, self.direction)
        # fill_val set to ``seg_ignore_label`` for the ignored value
        # of segmentation map.
        self._translate_seg(
            results, offset, self.direction, fill_val=self.seg_ignore_label)
        self._filter_invalid(results, min_size=self.min_size)
        return results


@PIPELINES.register_module()
class ColorTransform(object):
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get('img_fields', ['img']):
            # NOTE defaultly the image should be BGR format
            img = results[key]
            results[key] = mmcv.adjust_color(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_color_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class EqualizeTransform(object):
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.prob = prob

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.imequalize(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._imequalize(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'


@PIPELINES.register_module()
class BrightnessTransform(object):
    """Apply Brightness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Brightness transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_brightness_img(self, results, factor=1.0):
        """Adjust the brightness of image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_brightness(img,
                                                  factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_brightness_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class ContrastTransform(object):
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_contrast_img(self, results, factor=1.0):
        """Adjust the image contrast."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_contrast(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_contrast_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str
