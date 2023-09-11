# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]


def _fixed_scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don't need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):
    """Resize images & bbox & seg.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class FixScaleResize(Resize):
    """Compared to Resize, FixScaleResize fixes the scaling issue when
    `keep_ratio=true`."""

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio


@TRANSFORMS.register_module()
class ResizeShortestEdge(BaseTransform):
    """Resize the image and mask while keeping the aspect ratio unchanged.

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L130 # noqa:E501

    This transform attempts to scale the shorter edge to the given
    `scale`, as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer
    edge does not exceed `max_size`.

    Required Keys:
        - img
        - gt_seg_map (optional)
    Modified Keys:
        - img
        - img_shape
        - gt_seg_map (optional))
    Added Keys:
        - scale
        - scale_factor
        - keep_ratio

    Args:
        scale (Union[int, Tuple[int, int]]): The target short edge length.
            If it's tuple, will select the min value as the short edge length.
        max_size (int): The maximum allowed longest edge length.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 max_size: Optional[int] = None,
                 resize_type: str = 'Resize',
                 **resize_kwargs) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    def _get_output_shape(
            self, img: np.ndarray,
            short_edge_length: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if self.max_size and max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return new_w, new_h

    def transform(self, results: dict) -> dict:
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)


@TRANSFORMS.register_module()
class FixShapeResize(Resize):
    """Resize images & bbox & seg to the specified size.

    This transform resizes the input image according to ``width`` and
    ``height``. Bboxes, masks, and seg map are then resized
    with the same parameters.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        width (int): width for resizing.
        height (int): height for resizing.
            Defaults to None.
        pad_val (Number | dict[str, Number], optional): Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 pad_val: Union[Number, dict] = dict(img=0, seg=255),
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:
        assert width is not None and height is not None, (
            '`width` and'
            '`height` can not be `None`')

        self.width = width
        self.height = height
        self.scale = (width, height)

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border

        if keep_ratio is True:
            # padding to the fixed size when keep_ratio=True
            self.pad_transform = Pad(size=self.scale, pad_val=pad_val)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        img = results['img']
        h, w = img.shape[:2]
        if self.keep_ratio:
            scale_factor = min(self.width / w, self.height / h)
            results['scale_factor'] = (scale_factor, scale_factor)
            real_w, real_h = int(w * float(scale_factor) +
                                 0.5), int(h * float(scale_factor) + 0.5)
            img, scale_factor = mmcv.imrescale(
                results['img'], (real_w, real_h),
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['keep_ratio'] = self.keep_ratio
            results['scale'] = (real_w, real_h)
        else:
            results['scale'] = (self.width, self.height)
            results['scale_factor'] = (self.width / w, self.height / h)
            super()._resize_img(results)

        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        if self.keep_ratio:
            self.pad_transform(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, height={self.height}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):
    """Flip the image & bbox & mask & segmentation map. Added or Updated keys:
    flip, flip_direction, img, gt_bboxes, and gt_seg_map. There are 3 flip
    modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.


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

    - flip
    - flip_direction
    - homography_matrix


    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['img'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class RandomShift(BaseTransform):
    """Shift the image and box given shift pixels and probability.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignore_flags (bool) (optional)

    Args:
        prob (float): Probability of shifts. Defaults to 0.5.
        max_shift_px (int): The max pixels for shifting. Defaults to 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Defaults to 1.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_shift_px: int = 32,
                 filter_thr_px: int = 1) -> None:
        assert 0 <= prob <= 1
        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if self._random_prob() < self.prob:
            img_shape = results['img'].shape[:2]

            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            new_x = max(0, random_shift_x)
            ori_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            ori_y = max(0, -random_shift_y)

            # TODO: support mask and semantic segmentation maps.
            bboxes = results['gt_bboxes'].clone()
            bboxes.translate_([random_shift_x, random_shift_y])

            # clip border
            bboxes.clip_(img_shape)

            # remove invalid bboxes
            valid_inds = (bboxes.widths > self.filter_thr_px).numpy() & (
                bboxes.heights > self.filter_thr_px).numpy()
            # If the shift does not contain any gt-bbox area, skip this
            # image.
            if not valid_inds.any():
                return results
            bboxes = bboxes[valid_inds]
            results['gt_bboxes'] = bboxes
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            # shift img
            img = results['img']
            new_img = np.zeros_like(img)
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(random_shift_y)
            new_w = img_w - np.abs(random_shift_x)
            new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                = img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
            results['img'] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_shift_px={self.max_shift_px}, '
        repr_str += f'filter_thr_px={self.filter_thr_px})'
        return repr_str


@TRANSFORMS.register_module()
class Pad(MMCV_Pad):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_masks
    - gt_seg_map

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (width, height). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def _pad_masks(self, results: dict) -> None:
        """Pad masks according to ``results['pad_shape']``."""
        if results.get('gt_masks', None) is not None:
            pad_val = self.pad_val.get('masks', 0)
            pad_shape = results['pad_shape'][:2]
            results['gt_masks'] = results['gt_masks'].pad(
                pad_shape, pad_val=pad_val)

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        self._pad_masks(results)
        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks.

    The absolute ``crop_size`` is sampled based on ``crop_type`` and
    ``image_size``, then the cropped results are generated.

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
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_seg_map (optional)
    - gt_instances_ids (options, only used in MOT/VIS)

    Added Keys:

    - homography_matrix

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            (width, height).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          ``gt_bboxes`` corresponds to ``gt_labels`` and ``gt_masks``, and
          ``gt_bboxes_ignore`` corresponds to ``gt_labels_ignore`` and
          ``gt_masks_ignore``.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size: tuple,
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True) -> None:
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
            if crop_type == 'absolute_range':
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == 'absolute_range':
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'recompute_bbox={self.recompute_bbox}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class SegRescale(BaseTransform):
    """Rescale semantic segmentation maps.

    This transform rescale the ``gt_seg_map`` according to ``scale_factor``.

    Required Keys:

    - gt_seg_map

    Modified Keys:

    - gt_seg_map

    Args:
        scale_factor (float): The scale factor of the final output. Defaults
            to 1.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self, scale_factor: float = 1, backend: str = 'cv2') -> None:
        self.scale_factor = scale_factor
        self.backend = backend

    def transform(self, results: dict) -> dict:
        """Transform function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        if self.scale_factor != 1:
            results['gt_seg_map'] = mmcv.imrescale(
                results['gt_seg_map'],
                self.scale_factor,
                interpolation='nearest',
                backend=self.backend)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale_factor={self.scale_factor}, '
        repr_str += f'backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        mode = random.randint(2)
        brightness_flag = random.randint(2)
        contrast_flag = random.randint(2)
        saturation_flag = random.randint(2)
        hue_flag = random.randint(2)
        swap_flag = random.randint(2)
        delta_value = random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)
        swap_value = random.permutation(3)

        return (mode, brightness_flag, contrast_flag, saturation_flag,
                hue_flag, swap_flag, delta_value, alpha_value,
                saturation_value, hue_value, swap_value)

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        img = img.astype(np.float32)

        (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
         swap_flag, delta_value, alpha_value, saturation_value, hue_value,
         swap_value) = self._random_flags()

        # random brightness
        if brightness_flag:
            img += delta_value

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if mode == 1:
            if contrast_flag:
                img *= alpha_value

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value
            # For image(type=float32), after convert bgr to hsv by opencv,
            # valid saturation value range is [0, 1]
            if saturation_value > 1:
                img[..., 1] = img[..., 1].clip(0, 1)

        # random hue
        if hue_flag:
            img[..., 0] += hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= alpha_value

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_value]

        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_delta={self.brightness_delta}, '
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)}, '
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)}, '
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class Expand(BaseTransform):
    """Random expand the image & bboxes & masks & segmentation map.

    Randomly place the original image on a canvas of ``ratio`` x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Args:
        mean (sequence): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (sequence)): range of expand ratio.
        seg_ignore_label (int): label of ignore segmentation map.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean: Sequence[Number] = (0, 0, 0),
                 to_rgb: bool = True,
                 ratio_range: Sequence[Number] = (1, 4),
                 seg_ignore_label: int = None,
                 prob: float = 0.5) -> None:
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    @cache_randomness
    def _random_ratio(self) -> float:
        return random.uniform(self.min_ratio, self.max_ratio)

    @cache_randomness
    def _random_left_top(self, ratio: float, h: int,
                         w: int) -> Tuple[int, int]:
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        return left, top

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to expand images, bounding boxes, masks,
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes, masks, segmentation
                map expanded.
        """
        if self._random_prob() > self.prob:
            return results
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        h, w, c = img.shape
        ratio = self._random_ratio()
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left, top = self._random_left_top(ratio, h, w)
        expand_img[top:top + h, left:left + w] = img
        results['img'] = expand_img
        results['img_shape'] = expand_img.shape[:2]

        # expand bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].translate_([left, top])

        # expand masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segmentation map
        if results.get('gt_seg_map', None) is not None:
            gt_seg = results['gt_seg_map']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_seg_map'] = expand_gt_seg

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class MinIoURandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks & segmentation map, the cropped
    patches have minimum IoU requirement with original image & bboxes & masks.

    & segmentation map, the IoU threshold is randomly selected from min_ious.


    Required Keys:

    - img
    - img_shape
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


    Args:
        min_ious (Sequence[float]): minimum IoU threshold for all intersections
            with bounding boxes.
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """

    def __init__(self,
                 min_ious: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size: float = 0.3,
                 bbox_clip_border: bool = True) -> None:

        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _random_mode(self) -> Number:
        return random.choice(self.sample_mode)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to crop images and bounding boxes with minimum
        IoU constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """
        assert 'img' in results, '`img` is not found in results'
        assert 'gt_bboxes' in results, '`gt_bboxes` is not found in results'
        img = results['img']
        boxes = results['gt_bboxes']
        h, w, c = img.shape
        while True:
            mode = self._random_mode()
            self.mode = mode
            if mode == 1:
                return results

            min_iou = self.mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = boxes.overlaps(
                    HorizontalBoxes(patch.reshape(-1, 4).astype(np.float32)),
                    boxes).numpy().reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        centers = boxes.centers.numpy()
                        mask = ((centers[:, 0] > patch[0]) *
                                (centers[:, 1] > patch[1]) *
                                (centers[:, 0] < patch[2]) *
                                (centers[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    if results.get('gt_bboxes', None) is not None:
                        boxes = results['gt_bboxes']
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes.translate_([-patch[0], -patch[1]])
                        if self.bbox_clip_border:
                            boxes.clip_(
                                [patch[3] - patch[1], patch[2] - patch[0]])
                        results['gt_bboxes'] = boxes

                        # ignore_flags
                        if results.get('gt_ignore_flags', None) is not None:
                            results['gt_ignore_flags'] = \
                                results['gt_ignore_flags'][mask]

                        # labels
                        if results.get('gt_bboxes_labels', None) is not None:
                            results['gt_bboxes_labels'] = results[
                                'gt_bboxes_labels'][mask]

                        # mask fields
                        if results.get('gt_masks', None) is not None:
                            results['gt_masks'] = results['gt_masks'][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape[:2]

                # seg fields
                if results.get('gt_seg_map', None) is not None:
                    results['gt_seg_map'] = results['gt_seg_map'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class Corrupt(BaseTransform):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Required Keys:

    - img (np.uint8)


    Modified Keys:

    - img (np.uint8)


    Args:
        corruption (str): Corruption name.
        severity (int): The severity of corruption. Defaults to 1.
    """

    def __init__(self, corruption: str, severity: int = 1) -> None:
        self.corruption = corruption
        self.severity = severity

    def transform(self, results: dict) -> dict:
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class Albu(BaseTransform):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - img_shape (tuple)

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict, optional): Bbox_params for albumentation `Compose`
        keymap (dict, optional): Contains
            {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug. Defaults to False.
    """

    def __init__(self,
                 transforms: List[dict],
                 bbox_params: Optional[dict] = None,
                 keymap: Optional[dict] = None,
                 skip_img_without_anno: bool = False) -> None:
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        # TODO: gt_seg_map is not currently supported
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results, ori_masks = self._preprocess_results(results)
        results = self.aug(**results)
        results = self._postprocess_results(results, ori_masks)
        if results is None:
            return None
        # back to the original format
        results = self.mapper(results, self.keymap_back)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def _preprocess_results(self, results: dict) -> tuple:
        """Pre-processing results to facilitate the use of Albu."""
        if 'bboxes' in results:
            # to list of boxes
            if not isinstance(results['bboxes'], HorizontalBoxes):
                raise NotImplementedError(
                    'Albu only supports horizontal boxes now')
            bboxes = results['bboxes'].numpy()
            results['bboxes'] = [x for x in bboxes]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        ori_masks = None
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        return results, ori_masks

    def _postprocess_results(
            self,
            results: dict,
            ori_masks: Optional[Union[BitmapMasks,
                                      PolygonMasks]] = None) -> dict:
        """Post-processing Albu output."""
        # albumentations may return np.array or list on different versions
        if 'gt_bboxes_labels' in results and isinstance(
                results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(
                results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(
                results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(
                results['gt_ignore_flags'], dtype=bool)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)
            results['bboxes'] = HorizontalBoxes(results['bboxes'])

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    assert ori_masks is not None
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], ori_masks.height, ori_masks.width)

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None
            elif 'masks' in results:
                results['masks'] = ori_masks.__class__(results['masks'],
                                                       ori_masks.height,
                                                       ori_masks.width)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class RandomCenterCropPad(BaseTransform):
    """Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code:: text

                        output image

               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+

    There are 5 main areas in the figure:

    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.

    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.

    Train pipeline:

    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.

    Test pipeline:

    1. Compute output shape according to ``test_pad_mode``.
    2. Generate padding image with center matches the original image
       center.
    3. Initialize the padding image with pixel value equals to ``mean``.
    4. Copy the ``cropped area`` to padding image.

    Required Keys:

    - img (np.float32)
    - img_shape (tuple)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img (np.float32)
    - img_shape (tuple)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Args:
        crop_size (tuple, optional): expected size after crop, final size will
            computed according to ratio. Requires  (width, height)
            in train mode, and None in test mode.
        ratios (tuple, optional): random select a ratio from tuple and crop
            image to (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode. Defaults to (0.9, 1.0, 1.1).
        border (int, optional): max distance from center select area to image
            border. Only available in train mode. Defaults to 128.
        mean (sequence, optional): Mean values of 3 channels.
        std (sequence, optional): Std values of 3 channels.
        to_rgb (bool, optional): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
            Defaults to False.
        test_pad_mode (tuple, optional): padding method and padding shape
            value, only available in test mode. Default is using
            'logical_or' with 127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)

            Defaults to ('logical_or', 127).
        test_pad_add_pix (int): Extra padding pixel in test mode.
            Defaults to 0.
        bbox_clip_border (bool): Whether clip the objects outside
            the border of the image. Defaults to True.
    """

    def __init__(self,
                 crop_size: Optional[tuple] = None,
                 ratios: Optional[tuple] = (0.9, 1.0, 1.1),
                 border: Optional[int] = 128,
                 mean: Optional[Sequence] = None,
                 std: Optional[Sequence] = None,
                 to_rgb: Optional[bool] = None,
                 test_mode: bool = False,
                 test_pad_mode: Optional[tuple] = ('logical_or', 127),
                 test_pad_add_pix: int = 0,
                 bbox_clip_border: bool = True) -> None:
        if test_mode:
            assert crop_size is None, 'crop_size must be None in test mode'
            assert ratios is None, 'ratios must be None in test mode'
            assert border is None, 'border must be None in test mode'
            assert isinstance(test_pad_mode, (list, tuple))
            assert test_pad_mode[0] in ['logical_or', 'size_divisor']
        else:
            assert isinstance(crop_size, (list, tuple))
            assert crop_size[0] > 0 and crop_size[1] > 0, (
                'crop_size must > 0 in train mode')
            assert isinstance(ratios, (list, tuple))
            assert test_pad_mode is None, (
                'test_pad_mode must be None in train mode')

        self.crop_size = crop_size
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.test_mode = test_mode
        self.test_pad_mode = test_pad_mode
        self.test_pad_add_pix = test_pad_add_pix
        self.bbox_clip_border = bbox_clip_border

    def _get_border(self, border, size):
        """Get final border for the target size.

        This function generates a ``final_border`` according to image's shape.
        The area between ``final_border`` and ``size - final_border`` is the
        ``center range``. We randomly choose center from the ``center range``
        to avoid our random center is too close to original image's border.
        Also ``center range`` should be larger than 0.

        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.

        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = boxes.centers.numpy()
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
            center[:, 0] < patch[2]) * (
                center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.

        This function is equivalent to generating a blank image with ``size``
        as its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with ``mean pixel``.

        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]

        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of
                ``cropped_img`` to the original image area, [top, bottom,
                left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],
                          dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        gt_bboxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[1] * scale)
            new_w = int(self.crop_size[0] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                if len(gt_bboxes) == 0:
                    results['img'] = cropped_img
                    results['img_shape'] = cropped_img.shape[:2]
                    return results

                # if image do not have valid bbox, any crop patch is valid.
                mask = self._filter_boxes(patch, gt_bboxes)
                if not mask.any():
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape[:2]

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                gt_bboxes = gt_bboxes[mask]
                gt_bboxes.translate_([
                    cropped_center_x - left_w - x0,
                    cropped_center_y - top_h - y0
                ])
                if self.bbox_clip_border:
                    gt_bboxes.clip_([new_h, new_w])
                keep = gt_bboxes.is_inside([new_h, new_w]).numpy()
                gt_bboxes = gt_bboxes[keep]

                results['gt_bboxes'] = gt_bboxes

                # ignore_flags
                if results.get('gt_ignore_flags', None) is not None:
                    gt_ignore_flags = results['gt_ignore_flags'][mask]
                    results['gt_ignore_flags'] = \
                        gt_ignore_flags[keep]

                # labels
                if results.get('gt_bboxes_labels', None) is not None:
                    gt_labels = results['gt_bboxes_labels'][mask]
                    results['gt_bboxes_labels'] = gt_labels[keep]

                if 'gt_masks' in results or 'gt_seg_map' in results:
                    raise NotImplementedError(
                        'RandomCenterCropPad only supports bbox.')

                return results

    def _test_aug(self, results):
        """Around padding the original image without cropping.

        The padding mode and value are from ``test_pad_mode``.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        if self.test_pad_mode[0] in ['logical_or']:
            # self.test_pad_add_pix is only used for centernet
            target_h = (h | self.test_pad_mode[1]) + self.test_pad_add_pix
            target_w = (w | self.test_pad_mode[1]) + self.test_pad_add_pix
        elif self.test_pad_mode[0] in ['size_divisor']:
            divisor = self.test_pad_mode[1]
            target_h = int(np.ceil(h / divisor)) * divisor
            target_w = int(np.ceil(w / divisor)) * divisor
        else:
            raise NotImplementedError(
                'RandomCenterCropPad only support two testing pad mode:'
                'logical-or and size_divisor.')

        cropped_img, border, _ = self._crop_image_and_paste(
            img, [h // 2, w // 2], [target_h, target_w])
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape[:2]
        results['border'] = border
        return results

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        if self.test_mode:
            return self._test_aug(results)
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'test_mode={self.test_mode}, '
        repr_str += f'test_pad_mode={self.test_pad_mode}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class CutOut(BaseTransform):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        n_holes (int or tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [``n_holes[0]``, ``n_holes[1]``].
        cutout_shape (tuple[int, int] or list[tuple[int, int]], optional):
            The candidate shape of dropped regions. It can be
            ``tuple[int, int]`` to use a fixed cutout shape, or
            ``list[tuple[int, int]]`` to randomly choose shape
            from the list. Defaults to None.
        cutout_ratio (tuple[float, float] or list[tuple[float, float]],
            optional): The candidate ratio of dropped regions. It can be
            ``tuple[float, float]`` to use a fixed ratio or
            ``list[tuple[float, float]]`` to randomly choose ratio
            from the list. Please note that ``cutout_shape`` and
            ``cutout_ratio`` cannot be both given at the same time.
            Defaults to None.
        fill_in (tuple[float, float, float] or tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Defaults to (0, 0, 0).
    """

    def __init__(
        self,
        n_holes: Union[int, Tuple[int, int]],
        cutout_shape: Optional[Union[Tuple[int, int],
                                     List[Tuple[int, int]]]] = None,
        cutout_ratio: Optional[Union[Tuple[float, float],
                                     List[Tuple[float, float]]]] = None,
        fill_in: Union[Tuple[float, float, float], Tuple[int, int,
                                                         int]] = (0, 0, 0)
    ) -> None:

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Call function to drop some regions of image."""
        h, w, c = results['img'].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            results['img'][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


@TRANSFORMS.register_module()
class Mosaic(BaseTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class MixUp(BaseTransform):
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 max_iters: int = 15,
                 bbox_clip_border: bool = True) -> None:
        assert isinstance(img_scale, tuple)
        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            gt_bboxes_i = dataset[index]['gt_bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[1], self.dynamic_scale[0], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale[::-1],
                dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[1] / retrieve_img.shape[0],
                          self.dynamic_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)

        # remove outside bbox
        inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape[:2]
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class RandomAffine(BaseTransform):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix


@TRANSFORMS.register_module()
class YOLOXHSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def transform(self, results: dict) -> dict:
        img = results['img']
        hsv_gains = self._get_hsv_gains()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@TRANSFORMS.register_module()
class CopyPaste(BaseTransform):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Defaults to True.
        paste_by_box (bool): Whether use boxes as masks when masks are not
            available.
            Defaults to False.
    """

    def __init__(
        self,
        max_num_pasted: int = 100,
        bbox_occluded_thr: int = 10,
        mask_occluded_thr: int = 300,
        selected: bool = True,
        paste_by_box: bool = False,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = paste_by_box

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'
        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            if self.paste_by_box:
                warnings.warn('gt_masks is already contained in results, '
                              'so paste_by_box is disabled.')
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = self.get_gt_masks(results)
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_bboxes_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        results['gt_ignore_flags'] = selected_ignore_flags
        return results

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = src_results['gt_masks']
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(
            dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])
        ignore_flags = np.concatenate(
            [dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        return dst_results

    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}), '
        repr_str += f'paste_by_box={self.paste_by_box})'
        return repr_str


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """RandomErasing operation.

    Random Erasing randomly selects a rectangle region
    in an image and erases its pixels with random values.
    `RandomErasing <https://arxiv.org/abs/1708.04896>`_.

    Required Keys:

    - img
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:
    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        n_patches (int or tuple[int, int]): Number of regions to be dropped.
            If it is given as a tuple, number of patches will be randomly
            selected from the closed interval [``n_patches[0]``,
            ``n_patches[1]``].
        ratio (float or tuple[float, float]): The ratio of erased regions.
            It can be ``float`` to use a fixed ratio or ``tuple[float, float]``
            to randomly choose ratio from the interval.
        squared (bool): Whether to erase square region. Defaults to True.
        bbox_erased_thr (float): The threshold for the maximum area proportion
            of the bbox to be erased. When the proportion of the area where the
            bbox is erased is greater than the threshold, the bbox will be
            removed. Defaults to 0.9.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
    """

    def __init__(
        self,
        n_patches: Union[int, Tuple[int, int]],
        ratio: Union[float, Tuple[float, float]],
        squared: bool = True,
        bbox_erased_thr: float = 0.9,
        img_border_value: Union[int, float, tuple] = 128,
        mask_border_value: int = 0,
        seg_ignore_label: int = 255,
    ) -> None:
        if isinstance(n_patches, tuple):
            assert len(n_patches) == 2 and 0 <= n_patches[0] < n_patches[1]
        else:
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, tuple):
            assert len(ratio) == 2 and 0 <= ratio[0] < ratio[1] <= 1
        else:
            ratio = (ratio, ratio)

        self.n_patches = n_patches
        self.ratio = ratio
        self.squared = squared
        self.bbox_erased_thr = bbox_erased_thr
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label

    @cache_randomness
    def _get_patches(self, img_shape: Tuple[int, int]) -> List[list]:
        """Get patches for random erasing."""
        patches = []
        n_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)
        for _ in range(n_patches):
            if self.squared:
                ratio = np.random.random() * (self.ratio[1] -
                                              self.ratio[0]) + self.ratio[0]
                ratio = (ratio, ratio)
            else:
                ratio = (np.random.random() * (self.ratio[1] - self.ratio[0]) +
                         self.ratio[0], np.random.random() *
                         (self.ratio[1] - self.ratio[0]) + self.ratio[0])
            ph, pw = int(img_shape[0] * ratio[0]), int(img_shape[1] * ratio[1])
            px1, py1 = np.random.randint(0,
                                         img_shape[1] - pw), np.random.randint(
                                             0, img_shape[0] - ph)
            px2, py2 = px1 + pw, py1 + ph
            patches.append([px1, py1, px2, py2])
        return np.array(patches)

    def _transform_img(self, results: dict, patches: List[list]) -> None:
        """Random erasing the image."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['img'][py1:py2, px1:px2, :] = self.img_border_value

    def _transform_bboxes(self, results: dict, patches: List[list]) -> None:
        """Random erasing the bboxes."""
        bboxes = results['gt_bboxes']
        # TODO: unify the logic by using operators in BaseBoxes.
        assert isinstance(bboxes, HorizontalBoxes)
        bboxes = bboxes.numpy()
        left_top = np.maximum(bboxes[:, None, :2], patches[:, :2])
        right_bottom = np.minimum(bboxes[:, None, 2:], patches[:, 2:])
        wh = np.maximum(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (
            bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.bbox_erased_thr
        results['gt_bboxes'] = HorizontalBoxes(bboxes[valid_inds])
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'][valid_inds]

    def _transform_masks(self, results: dict, patches: List[list]) -> None:
        """Random erasing the masks."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['gt_masks'].masks[:, py1:py2,
                                      px1:px2] = self.mask_border_value

    def _transform_seg(self, results: dict, patches: List[list]) -> None:
        """Random erasing the segmentation map."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['gt_seg_map'][py1:py2, px1:px2] = self.seg_ignore_label

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to erase some regions of image."""
        patches = self._get_patches(results['img_shape'])
        self._transform_img(results, patches)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results, patches)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results, patches)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results, patches)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_patches={self.n_patches}, '
        repr_str += f'ratio={self.ratio}, '
        repr_str += f'squared={self.squared}, '
        repr_str += f'bbox_erased_thr={self.bbox_erased_thr}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@TRANSFORMS.register_module()
class CachedMosaic(Mosaic):
    """Cached mosaic augmentation.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The cached mosaic transform steps are as follows:

         1. Append the results from the last transform into the cache.
         2. Choose the mosaic center as the intersections of 4 images
         3. Get the left top image according to the index, and randomly
            sample another 3 images from the result cache.
         4. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results_cache = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, 'The length of cache must >= 4, ' \
                                       f'but got {max_cached_images}.'
        self.max_cached_images = max_cached_images

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if 'gt_masks' in results else False

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal')
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        if with_mask:
            mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
            results['gt_masks'] = mosaic_masks[inside_inds]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop})'
        return repr_str


@TRANSFORMS.register_module()
class CachedMixUp(BaseTransform):
    """Cached mixup data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The cached mixup transform steps are as follows:

        1. Append the results from the last transform into the cache.
        2. Another random image is picked from the cache and embedded in
           the top left patch(after padding and resizing)
        3. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 max_iters: int = 15,
                 bbox_clip_border: bool = True,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                       f'but got {max_cached_images}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.results_cache = []

        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = prob

    @cache_randomness
    def get_indexes(self, cache: list) -> int:
        """Call function to collect indexes.

        Args:
            cache (list): The result cache.

        Returns:
            int: index.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(cache) - 1)
            gt_bboxes_i = cache[index]['gt_bboxes']
            if len(gt_bboxes_i) != 0:
                break
        return index

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return results

        if random.uniform(0, 1) > self.prob:
            return results

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO: refactor mixup to reuse these code.
        if retrieve_results['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_img = retrieve_results['img']
        with_mask = True if 'gt_masks' in results else False

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[1], self.dynamic_scale[0], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale[::-1],
                dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[1] / retrieve_img.shape[0],
                          self.dynamic_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if with_mask:
            retrieve_gt_masks = retrieve_results['gt_masks'].rescale(
                scale_ratio)

        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')
            if with_mask:
                retrieve_gt_masks = retrieve_gt_masks.flip()

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if with_mask:
            retrieve_gt_masks = retrieve_gt_masks.translate(
                out_shape=(target_h, target_w),
                offset=-x_offset,
                direction='horizontal')
            retrieve_gt_masks = retrieve_gt_masks.translate(
                out_shape=(target_h, target_w),
                offset=-y_offset,
                direction='vertical')

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        if with_mask:
            mixup_gt_masks = retrieve_gt_masks.cat(
                [results['gt_masks'], retrieve_gt_masks])

        # remove outside bbox
        inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]
        if with_mask:
            mixup_gt_masks = mixup_gt_masks[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape[:2]
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags
        if with_mask:
            results['gt_masks'] = mixup_gt_masks
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop}, '
        repr_str += f'prob={self.prob})'
        return repr_str
