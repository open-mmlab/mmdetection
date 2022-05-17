# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def create_random_bboxes(num_bboxes, img_w, img_h):
    bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
    bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
    bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
    bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
        np.float32)
    return bboxes
