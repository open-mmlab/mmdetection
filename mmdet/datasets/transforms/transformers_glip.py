# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from .transforms import RandomFlip


@TRANSFORMS.register_module()
class GTBoxSubOne_GLIP(BaseTransform):
    """Subtract 1 from the x2 and y2 coordinates of the gt_bboxes."""

    def transform(self, results: dict) -> dict:
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            if isinstance(gt_bboxes, np.ndarray):
                gt_bboxes[:, 2:] -= 1
                results['gt_bboxes'] = gt_bboxes
            elif isinstance(gt_bboxes, HorizontalBoxes):
                gt_bboxes = results['gt_bboxes'].tensor
                gt_bboxes[:, 2:] -= 1
                results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
            else:
                raise NotImplementedError
        return results


@TRANSFORMS.register_module()
class RandomFlip_GLIP(RandomFlip):
    """Flip the image & bboxes & masks & segs horizontally or vertically.

    When using horizontal flipping, the corresponding bbox x-coordinate needs
    to be additionally subtracted by one.
    """

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
            # Only change this line
            if results['flip_direction'] == 'horizontal':
                results['gt_bboxes'].translate_([-1, 0])

        # TODO: check it
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
