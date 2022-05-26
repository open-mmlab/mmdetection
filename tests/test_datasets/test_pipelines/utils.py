# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.core.mask import BitmapMasks


def create_random_bboxes(num_bboxes, img_w, img_h):
    bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
    bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
    bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
    bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
        np.float32)
    return bboxes


def create_full_masks(gt_bboxes, img_w, img_h):
    xmin, ymin = gt_bboxes[:, 0:1], gt_bboxes[:, 1:2]
    xmax, ymax = gt_bboxes[:, 2:3], gt_bboxes[:, 3:4]
    gt_masks = np.zeros((len(gt_bboxes), img_h, img_w), dtype=np.uint8)
    for i in range(len(gt_bboxes)):
        gt_masks[i, int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])] = 1
    gt_masks = BitmapMasks(gt_masks, img_h, img_w)
    return gt_masks
