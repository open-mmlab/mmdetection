import cv2
import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def random_negative(value, random_negative_prob):
    """Randomly negative value based on random_negative_prob."""
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
class Shear(object):
    """Apply Shear Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level for Translate and should be in
            range (0,_MAX_LEVEL]. This value controls the offset used for
            translate the image/bboxes/masks/seg along with x-axis or y-axis.
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The ``img_fill_val`` used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform translating and should be in
            range 0 to 1.
        axis (str): Translate images along with x-axis or y-axis. The option
            of axis is 'x' or 'y'.
        max_shear_magnitude (float): The maximum magnitude for shear
            transformation.
    """

    def __init__(self,
                 level,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 axis='x',
                 max_shear_magnitude=0.3,
                 *args,
                 **kwargs):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range (0,_MAX_LEVEL]'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range 0 to 1.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)])
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, \
                'img_fill_val as tuple must have 3 elements.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255].'
        assert axis in ('x', 'y'), \
            'Translate should be alone with x-axis or y-axis.'
        assert isinstance(max_shear_magnitude, float), \
            'max_shear_magnitude should be type float.'
        assert 0. <= max_shear_magnitude < 1., \
            'Defaultly max_shear_magnitude should be in range [0,1).'
        self.level = level
        self.magnitude = level_to_value(level, max_shear_magnitude)
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.axis = axis
        self.max_shear_magnitude = max_shear_magnitude

    @staticmethod
    def warpAffine(data,
                   trans_matrix,
                   out_size,
                   fill_val,
                   flags=cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT):
        """Affine wrapper which transforms the source data using the given
        trans_matrix.

        Args:
            data (np.ndarray): Source data.
            trans_matrix (np.ndarray): Transformation matrix with shape (2, 3).
            out_size (tuple): Size of the output data with format (w, h).
            fill_val (int | float | tuple): Value used in case of a constant
                border.
            flags: Interpolation methods used in ``cv2.warpAffine``.
            borderMode: pixel extrapolation method used in ``cv2.warpAffine``.

        Returns:
            np.ndarray: transformed data with the same shape as input data.
        """
        return cv2.warpAffine(
            data,
            trans_matrix,
            dsize=out_size,  # dsize takes input size as order (w,h).
            flags=flags,
            borderMode=borderMode,
            borderValue=fill_val)

    @staticmethod
    def _get_shear_matrix(magnitude, axis='x'):
        """Generates the transformation matrix for shear augmentation."""
        if axis == 'x':
            shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
        elif axis == 'y':
            shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
        return shear_matrix

    def _shear_img(self, results, shear_matrix, fill_val):
        """Shear the image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self.warpAffine(img, shear_matrix,
                                           img.shape[:2][::-1],
                                           fill_val).astype(img.dtype)

    def _shear_bboxes(self, results, magnitude):
        """Shear the bboxes."""
        h, w, c = results['img_shape']
        if self.axis == 'x':
            shear_matrix = np.stack([[1, 0], [-magnitude,
                                              1]]).astype(np.float32)  # [2, 2]
        else:
            shear_matrix = np.stack([[1, -magnitude], [0,
                                                       1]]).astype(np.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_y, min_x], [min_y, max_x],
                                    [max_y, min_x],
                                    [max_y, max_x]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose(
                (2, 1, 0)).astype(np.float32)  # [nb_box, 2, 4]
            new_coords = np.matmul(shear_matrix[None, :, :],
                                   coordinates)  # [nb_box, 2, 4]
            min_x, min_y = np.min(
                new_coords[:, 1, :], axis=-1), np.min(
                    new_coords[:, 0, :], axis=-1)
            max_x, max_y = np.max(
                new_coords[:, 1, :], axis=-1), np.max(
                    new_coords[:, 0, :], axis=-1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _shear_masks(self, results, shear_matrix, fill_val=0):
        """Shear the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            if isinstance(results[key], PolygonMasks):
                raise NotImplementedError
            elif isinstance(results[key], BitmapMasks):
                masks = results[key]
                results[key] = masks.shear(
                    shear_matrix, out_shape=(h, w), fill_val=fill_val)

    def _shear_seg(self, results, shear_matrix, fill_val=255):
        """Shear the segmentation maps."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            results[key] = self.warpAffine(seg, shear_matrix,
                                           seg.shape[:2][::-1],
                                           fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        # TODO check whether need or not?
        """Filter bboxes and corresponding masks too small after shear
        augmentation."""
        # The key correspondence from bboxes to labels and masks.
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results, random_negative_prob=0.5):
        """Call function to shear images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            random_negative_prob (float): The probability that turns the
             offset negative.

        Returns:
            dict: Sheared results.
        """
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, random_negative_prob)
        # the shear matrix used for transformation
        shear_matrix = self._get_shear_matrix(magnitude, self.axis)
        self._shear_img(results, shear_matrix, fill_val=self.img_fill_val)
        self._shear_bboxes(results, magnitude)
        # fill_val set to 0 for background of mask.
        self._shear_masks(results, shear_matrix, fill_val=0)
        self._shear_seg(results, shear_matrix, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'axis={self.axis}, '
        repr_str += f'max_shear_magnitude={self.max_shear_magnitude})'
        return repr_str
