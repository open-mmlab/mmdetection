import mmcv
import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

_MAX_LEVEL = 10


def level_to_value(level, max_value, random_negative_prob=0.5):
    """Mapping level to value based on _MAX_LEVEL and max_value."""
    return (level / _MAX_LEVEL) * max_value


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and segmentation
    maps."""
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
    """

    def __init__(
        self,
        level,
        prob=0.5,
        img_fill_val=128,
        seg_ignore_label=255,
        direction='horizontal',
        max_translate_offset=250.,
    ):
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

    def _translate_img(self,
                       results,
                       offset,
                       direction='horizontal',
                       interpolation='bilinear'):
        """Translate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            interpolation (str): Same as :func:`mmcv.imtranslate`.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(img, offset, direction,
                                            self.img_fill_val,
                                            interpolation).astype(img.dtype)

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

            # the boxs translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y],
                                          axis=-1)

    def _translate_masks(self,
                         results,
                         offset,
                         direction='horizontal',
                         fill_val=0,
                         interpolation='bilinear'):
        """Translate masks horizontally or vertically."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            if isinstance(masks, PolygonMasks):
                # PolygonMasks is not supported currently
                raise NotImplementedError
            elif isinstance(masks, BitmapMasks):
                results[key] = masks.translate((h, w), offset, direction,
                                               fill_val, interpolation)

    def _translate_seg(self,
                       results,
                       offset,
                       direction='horizontal',
                       fill_val=255,
                       interpolation='bilinear'):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction, fill_val,
                                            interpolation).astype(seg.dtype)

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
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

    def __call__(self,
                 results,
                 random_negative_prob=0.5,
                 interpolation='bilinear',
                 min_size=0.0):
        """Call function to translate images, bounding boxes, masks and
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            random_negative_prob (float): The probability that turns the
                offset negative.
            interpolation (str): Same as :func:`mmcv.imtranslate`.
            min_size (int | float): The minimum pixel for filtering
                invalid bboxes.

        Returns:
            dict: Translated results.
        """
        if np.random.rand() > self.prob:
            return results
        offset = random_negative(self.offset, random_negative_prob)
        self._translate_img(results, offset, self.direction, interpolation)
        self._translate_bboxes(results, offset)
        # fill_val set to 0 for background of mask.
        self._translate_masks(
            results,
            offset,
            self.direction,
            fill_val=0,
            interpolation=interpolation)
        # fill_val set to ``seg_ignore_label`` for the ignored value
        # of segmentation map.
        self._translate_seg(
            results,
            offset,
            self.direction,
            fill_val=self.seg_ignore_label,
            interpolation=interpolation)
        self._filter_invalid(results, min_size=min_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'max_translate_offset={self.max_translate_offset})'
        return repr_str
