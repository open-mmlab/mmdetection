import numpy as np
import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

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
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def batch_bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Similar to `bbox_overlaps` but the input boxes has an extra batch dim B.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (B, m, n) if is_aligned == False else shape
            (B, m, 1)
    """

    assert mode in ['iou', 'iof']

    B, rows, _ = bboxes1.size()
    _, cols, _ = bboxes2.size()
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(B, rows, 1) if is_aligned else bboxes1.new(
            B, rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, :, None, :2], bboxes2[:, None, :, :2])
        rb = torch.min(bboxes1[:, :, None, 2:], bboxes2[:, None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, :, :, 0] * wh[:, :, :, 1]
        area1 = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (
            bboxes1[:, :, 3] - bboxes1[:, :, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (
                bboxes2[:, :, 3] - bboxes2[:, :, 1])
            ious = overlap / (area1[:, :, None] + area2[:, None, :] - overlap)
        else:
            ious = overlap / (area1[:, :, None])

    return ious
