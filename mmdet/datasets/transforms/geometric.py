# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type
from .augment_wrappers import _MAX_LEVEL, level_to_mag


@TRANSFORMS.register_module()
class GeomTransform(BaseTransform):
    """Base class for geometric transformations. All geometric transformations
    need to inherit from this base class. ``GeomTransform`` unifies the class
    attributes and class functions of geometric transformations (ShearX,
    ShearY, Rotate, TranslateX, and TranslateY), and records the homography
    matrix.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for geometric transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for geometric transformation.
            Defaults to 1.0.
        reversal_prob (float): The probability that reverses the geometric
            transformation magnitude. Should be in range [0,1].
            Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 1.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        assert isinstance(reversal_prob, float), \
            f'reversal_prob should be type float, got {type(max_mag)}.'
        assert 0 <= reversal_prob <= 1.0, \
            f'The reversal probability of the transformation magnitude ' \
            f'should be type float, got {type(reversal_prob)}.'
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_border_value]), 'all ' \
            'elements of img_border_value should between range [0,255].' \
            f'got {img_border_value}.'
        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.reversal_prob = reversal_prob
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _transform_img(self, results: dict, mag: float) -> None:
        """Transform the image."""
        pass

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Transform the masks."""
        pass

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Transform the segmentation map."""
        pass

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for the geometric transformation."""
        return np.eye(3, dtype=np.float32)

    def _transform_bboxes(self, results: dict, mag: float) -> None:
        """Transform the bboxes."""
        results['gt_bboxes'].project_(self.homography_matrix)
        results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the geometric transformation."""
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = self.homography_matrix
        else:
            results['homography_matrix'] = self.homography_matrix @ results[
                'homography_matrix']

    @cache_randomness
    def _random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        return -mag if np.random.rand() > self.reversal_prob else mag

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function for images, bounding boxes, masks and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """

        if self._random_disable():
            return results
        mag = self._get_mag()
        self.homography_matrix = self._get_homography_matrix(results, mag)
        self._record_homography_matrix(results)
        self._transform_img(results, mag)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results, mag)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results, mag)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results, mag)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag}, '
        repr_str += f'reversal_prob={self.reversal_prob}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class ShearX(GeomTransform):
    """Shear the images, bboxes, masks and segmentation map horizontally.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing Shear and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum angle for the horizontal shear.
            Defaults to 0.0.
        max_mag (float): The maximum angle for the horizontal shear.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the horizontal
            shear magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 30.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 90., \
            f'min_mag angle for ShearX should be ' \
            f'in range [0, 90], got {min_mag}.'
        assert 0. <= max_mag <= 90., \
            f'max_mag angle for ShearX should be ' \
            f'in range [0, 90], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        mag = np.tan(mag * np.pi / 180)
        return -mag if np.random.rand() > self.reversal_prob else mag

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for ShearX."""
        return np.array([[1, mag, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Shear the image horizontally."""
        results['img'] = mmcv.imshear(
            results['img'],
            mag,
            direction='horizontal',
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Shear the masks horizontally."""
        results['gt_masks'] = results['gt_masks'].shear(
            results['img_shape'],
            mag,
            direction='horizontal',
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Shear the segmentation map horizontally."""
        results['gt_seg_map'] = mmcv.imshear(
            results['gt_seg_map'],
            mag,
            direction='horizontal',
            border_value=self.seg_ignore_label,
            interpolation='nearest')


@TRANSFORMS.register_module()
class ShearY(GeomTransform):
    """Shear the images, bboxes, masks and segmentation map vertically.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing ShearY and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum angle for the vertical shear.
            Defaults to 0.0.
        max_mag (float): The maximum angle for the vertical shear.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the vertical
            shear magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 30.,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 90., \
            f'min_mag angle for ShearY should be ' \
            f'in range [0, 90], got {min_mag}.'
        assert 0. <= max_mag <= 90., \
            f'max_mag angle for ShearY should be ' \
            f'in range [0, 90], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        mag = np.tan(mag * np.pi / 180)
        return -mag if np.random.rand() > self.reversal_prob else mag

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for ShearY."""
        return np.array([[1, 0, 0], [mag, 1, 0], [0, 0, 1]], dtype=np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Shear the image vertically."""
        results['img'] = mmcv.imshear(
            results['img'],
            mag,
            direction='vertical',
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Shear the masks vertically."""
        results['gt_masks'] = results['gt_masks'].shear(
            results['img_shape'],
            mag,
            direction='vertical',
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Shear the segmentation map vertically."""
        results['gt_seg_map'] = mmcv.imshear(
            results['gt_seg_map'],
            mag,
            direction='vertical',
            border_value=self.seg_ignore_label,
            interpolation='nearest')


@TRANSFORMS.register_module()
class Rotate(GeomTransform):
    """Rotate the images, bboxes, masks and segmentation map.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The maximum angle for rotation.
            Defaults to 0.0.
        max_mag (float): The maximum angle for rotation.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the rotation
            magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 30.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 180., \
            f'min_mag for Rotate should be in range [0,180], got {min_mag}.'
        assert 0. <= max_mag <= 180., \
            f'max_mag for Rotate should be in range [0,180], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for Rotate."""
        img_shape = results['img_shape']
        center = ((img_shape[1] - 1) * 0.5, (img_shape[0] - 1) * 0.5)
        cv2_rotation_matrix = cv2.getRotationMatrix2D(center, -mag, 1.0)
        return np.concatenate(
            [cv2_rotation_matrix,
             np.array([0, 0, 1]).reshape((1, 3))]).astype(np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Rotate the image."""
        results['img'] = mmcv.imrotate(
            results['img'],
            mag,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Rotate the masks."""
        results['gt_masks'] = results['gt_masks'].rotate(
            results['img_shape'],
            mag,
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Rotate the segmentation map."""
        results['gt_seg_map'] = mmcv.imrotate(
            results['gt_seg_map'],
            mag,
            border_value=self.seg_ignore_label,
            interpolation='nearest')


@TRANSFORMS.register_module()
class TranslateX(GeomTransform):
    """Translate the images, bboxes, masks and segmentation map horizontally.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum pixel's offset ratio for horizontal
            translation. Defaults to 0.0.
        max_mag (float): The maximum pixel's offset ratio for horizontal
            translation. Defaults to 0.1.
        reversal_prob (float): The probability that reverses the horizontal
            translation magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 0.1,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 1., \
            f'min_mag ratio for TranslateX should be ' \
            f'in range [0, 1], got {min_mag}.'
        assert 0. <= max_mag <= 1., \
            f'max_mag ratio for TranslateX should be ' \
            f'in range [0, 1], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for TranslateX."""
        mag = int(results['img_shape'][1] * mag)
        return np.array([[1, 0, mag], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Translate the image horizontally."""
        mag = int(results['img_shape'][1] * mag)
        results['img'] = mmcv.imtranslate(
            results['img'],
            mag,
            direction='horizontal',
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Translate the masks horizontally."""
        mag = int(results['img_shape'][1] * mag)
        results['gt_masks'] = results['gt_masks'].translate(
            results['img_shape'],
            mag,
            direction='horizontal',
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Translate the segmentation map horizontally."""
        mag = int(results['img_shape'][1] * mag)
        results['gt_seg_map'] = mmcv.imtranslate(
            results['gt_seg_map'],
            mag,
            direction='horizontal',
            border_value=self.seg_ignore_label,
            interpolation='nearest')


@TRANSFORMS.register_module()
class TranslateY(GeomTransform):
    """Translate the images, bboxes, masks and segmentation map vertically.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum pixel's offset ratio for vertical
            translation. Defaults to 0.0.
        max_mag (float): The maximum pixel's offset ratio for vertical
            translation. Defaults to 0.1.
        reversal_prob (float): The probability that reverses the vertical
            translation magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 0.1,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 1., \
            f'min_mag ratio for TranslateY should be ' \
            f'in range [0,1], got {min_mag}.'
        assert 0. <= max_mag <= 1., \
            f'max_mag ratio for TranslateY should be ' \
            f'in range [0,1], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for TranslateY."""
        mag = int(results['img_shape'][0] * mag)
        return np.array([[1, 0, 0], [0, 1, mag], [0, 0, 1]], dtype=np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Translate the image vertically."""
        mag = int(results['img_shape'][0] * mag)
        results['img'] = mmcv.imtranslate(
            results['img'],
            mag,
            direction='vertical',
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Translate masks vertically."""
        mag = int(results['img_shape'][0] * mag)
        results['gt_masks'] = results['gt_masks'].translate(
            results['img_shape'],
            mag,
            direction='vertical',
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Translate segmentation map vertically."""
        mag = int(results['img_shape'][0] * mag)
        results['gt_seg_map'] = mmcv.imtranslate(
            results['gt_seg_map'],
            mag,
            direction='vertical',
            border_value=self.seg_ignore_label,
            interpolation='nearest')
