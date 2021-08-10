import numpy as np


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6, extra_length=0.):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): shape (n, 4)
        bboxes2 (ndarray): shape (k, 4)
        mode (str): iou (intersection over union) or iof (intersection
            over foreground)
        extra_length (float): `extra_length` should be added when calculate
            the width and height, which means w, h should be computed as
            'x2 - x1 + extra_length` and 'y2 - y1 + extra_length'.
            Default: 0.
    Returns:
        ious (ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']
    assert extra_length in (0.,
                            1.), 'Only allow the `extra_length` to be 0 or 1'
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
