import cv2
import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES

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
class Rotate(object):
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Rotation angle in degrees. Positive values
            mean counter-clockwise rotation. Same in
            ``cv2.getRotationMatrix2D``.
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum degrees for rotate
            transformation.
    """

    def __init__(self,
                 level,
                 scale=1,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 max_rotate_angle=30,
                 *args,
                 **kwargs):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range (0,_MAX_LEVEL]'
        assert isinstance(scale, (int, float)), \
            'The scale must be type int or float.'
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
        assert isinstance(max_rotate_angle, (int, float)), \
            'max_rotate_angle should be type int or float.'
        self.level = level
        self.scale = scale
        self.degrees = level_to_value(level, max_rotate_angle)
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle

    @staticmethod
    def get_rotate_matrix(center, angle, scale):
        """Generates the rotate matrix used for ``cv2.warpAffine``.
        Args:
            center (tuple): Tuple with format (w, h). Center of the rotation
                in the source data.
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation (the coordinate origin is
                assumed to be the top-left corner).
            scale (int | float): Isotropic scale factor. Same in
                ``cv2.getRotationMatrix2D``.

        Returns:
            np.ndarray: The output affine transformation, 2x3
                floating-point matrix.
        """
        return cv2.getRotationMatrix2D(center, angle, scale)

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

    def _rotate_img(self, results, rotate_matrix, fill_val=128):
        """Rotate image based on the rotate_matrix."""
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = self.warpAffine(img, rotate_matrix,
                                           img.shape[:2][::-1],
                                           fill_val).astype(img.dtype)

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
            rotate_matrix = rotate_matrix.copy()  # [2, 3]
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

    def _rotate_masks(self, results, rotate_matrix, fill_val=0):
        """Rotate the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            if isinstance(masks, PolygonMasks):
                raise NotImplementedError
            elif isinstance(masks, BitmapMasks):
                results[key] = masks.rotate(rotate_matrix, (h, w), fill_val)

    def _rotate_seg(self, results, rotate_matrix, fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            results[key] = self.warpAffine(seg, rotate_matrix,
                                           seg.shape[:2][::-1],
                                           fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
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
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            random_negative_prob (float): The probability that turns the
             offset negative.
        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        degrees = random_negative(self.degrees, random_negative_prob)
        h, w, c = results['img_shape']
        # the rotate matrix used for transformation, defaultly
        # the image center without shift is used.
        rotate_matrix = self.get_rotate_matrix(
            center=(w / 2, h / 2), angle=degrees, scale=self.scale)
        self._rotate_img(results, rotate_matrix, fill_val=self.img_fill_val)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, rotate_matrix, fill_val=0)
        self._rotate_seg(
            results, rotate_matrix, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_rotate_angle={self.max_rotate_angle})'
        return repr_str
