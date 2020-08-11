"""
CommandLine:
    pytest tests/test_soft_nms.py
"""
import numpy as np
import torch

from mmdet.ops.nms.nms_wrapper import soft_nms


def test_soft_nms_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_soft_nms.py test_soft_nms_device_and_dtypes_cpu
    """
    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.FloatTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = base_dets.astype(np.float64)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.DoubleTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4
