# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

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


@PIPELINES.register_module()
class Resize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class RandomShift:
    """Shift the image and box given shift pixels and probability.

    Args:
        shift_ratio (float): Probability of shifts. Default 0.5.
        max_shift_px (int): The max pixels for shifting. Default 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Default 1.
    """

    def __init__(self, shift_ratio=0.5, max_shift_px=32, filter_thr_px=1):
        assert 0 <= shift_ratio <= 1
        assert max_shift_px >= 0
        self.shift_ratio = shift_ratio
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        # The key correspondence from bboxes to labels.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        """Call function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if random.random() < self.shift_ratio:
            img_shape = results['img'].shape[:2]

            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            new_x = max(0, random_shift_x)
            orig_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            orig_y = max(0, -random_shift_y)

            # TODO: support mask and semantic segmentation maps.
            for key in results.get('bbox_fields', []):
                bboxes = results[key].copy()
                bboxes[..., 0::2] += random_shift_x
                bboxes[..., 1::2] += random_shift_y

                # clip border
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, img_shape[1])
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, img_shape[0])

                # remove invalid bboxes
                bbox_w = bboxes[..., 2] - bboxes[..., 0]
                bbox_h = bboxes[..., 3] - bboxes[..., 1]
                valid_inds = (bbox_w > self.filter_thr_px) & (
                    bbox_h > self.filter_thr_px)
                # If the shift does not contain any gt-bbox area, skip this
                # image.
                if key == 'gt_bboxes' and not valid_inds.any():
                    return results
                bboxes = bboxes[valid_inds]
                results[key] = bboxes

                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            for key in results.get('img_fields', ['img']):
                img = results[key]
                new_img = np.zeros_like(img)
                img_h, img_w = img.shape[:2]
                new_h = img_h - np.abs(random_shift_y)
                new_w = img_w - np.abs(random_shift_x)
                new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                    = img[orig_y:orig_y + new_h, orig_x:orig_x + new_w]
                results[key] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_shift_px={self.max_shift_px}, '
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class SegRescale:
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self, scale_factor=1, backend='cv2'):
        self.scale_factor = scale_factor
        self.backend = backend

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """

        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key],
                    self.scale_factor,
                    interpolation='nearest',
                    backend=self.backend)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion:
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

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        img = img.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class Expand:
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        # expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@PIPELINES.register_module()
class MinIoURandomCrop:
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
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
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        if self.bbox_clip_border:
                            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class Corrupt:
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

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
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
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
        self.update_pad_shape = update_pad_shape
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

    def albu_builder(self, cfg):
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
        if mmcv.is_str(obj_type):
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
    def mapper(d, keymap):
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

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomCenterCropPad:
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

    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)
        test_pad_add_pix (int): Extra padding pixel in test mode. Default 0.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """

    def __init__(self,
                 crop_size=None,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 test_mode=False,
                 test_pad_mode=('logical_or', 127),
                 test_pad_add_pix=0,
                 bbox_clip_border=True):
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
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
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
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    if self.bbox_clip_border:
                        bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                        bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                        bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomCenterCropPad only supports bbox.')

                # crop semantic seg
                for key in results.get('seg_fields', []):
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
        results['img_shape'] = img.shape
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
        results['pad_shape'] = cropped_img.shape
        results['border'] = border
        return results

    def __call__(self, results):
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


@PIPELINES.register_module()
class CutOut:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

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

    def __call__(self, results):
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


@PIPELINES.register_module()
class Mosaic:
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

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
           image. Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
           output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 skip_filter=True,
                 pad_val=114):
        assert isinstance(img_scale, tuple)
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.skip_filter = skip_filter
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        results = self._mosaic_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
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
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
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
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                             2 * self.img_scale[1])
            mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                             2 * self.img_scale[0])
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
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
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str


@PIPELINES.register_module()
class MixUp:
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

     The mixup transform steps are as follows::

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
           Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
           Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
           Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
           iterations is greater than `max_iters`, but gt_bbox is still
           empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 skip_filter=True):
        assert isinstance(img_scale, tuple)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.skip_filter = skip_filter

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """

        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index

    def _mixup_transform(self, results):
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
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
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
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
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
        retrieve_gt_bboxes[:, 0::2] = np.clip(
            retrieve_gt_bboxes[:, 0::2] * scale_ratio, 0, origin_w)
        retrieve_gt_bboxes[:, 1::2] = np.clip(
            retrieve_gt_bboxes[:, 1::2] * scale_ratio, 0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset, 0, target_w)
        cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset, 0, target_h)

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

            if keep_list.sum() >= 1.0:
                retrieve_gt_labels = retrieve_gt_labels[keep_list]
                cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels

        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 max_rotate_degree=10.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 skip_filter=True):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.skip_filter = skip_filter

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

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

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack(
                    (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                if not self.skip_filter:
                    # filter bboxes
                    valid_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes)
                    results[key] = warp_bboxes[valid_index]
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            results['gt_labels'] = results['gt_labels'][
                                valid_index]
                else:
                    results[key] = warp_bboxes

                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'RandomAffine only supports bbox.')
        return results

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix


@PIPELINES.register_module()
class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Args:
        hue_delta (int): delta of hue. Default: 5.
        saturation_delta (int): delta of saturation. Default: 30.
        value_delta (int): delat of value. Default: 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results):
        img = results['img']
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
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
