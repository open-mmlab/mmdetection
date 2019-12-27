import torch

from .registry import IOU_CALCULATORS


@IOU_CALCULATORS.register_module
class BboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mode={}, is_aligned={})'.format(self.mode,
                                                      self.is_aligned)
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
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
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.
        Args:
            bboxes (Tensor): shape (m, 4)
            scale (float): the scale factor of bboxes

        Returns:
            (Tensor): shape (m, 4) scaled bboxes
        """
    w_half = (bboxes[:, 2] - bboxes[:, 0] + 1) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1] + 1) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0] + 1) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1] + 1) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(bboxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half - 1
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half - 1
    return boxes_exp


def is_located_in(points, bboxes, is_aligned=False):
    """ is center a locates in box b
    Then we compute the area of intersect between box_a and box_b.
    Args:
      points: (tensor) bounding boxes, Shape: [m,2].
      bboxes: (tensor)  bounding boxes, Shape: [n,4].
       If is_aligned is ``True``, then m mush be equal to n
    Return:
      (tensor) intersection area, Shape: [m, n]. If is_aligned ``True``,
       then shape = [m]
    """
    if not is_aligned:
        return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
               (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))
    else:
        return (points[:, 0] > bboxes[:, 0]) & \
               (points[:, 0] < bboxes[:, 2]) & \
               (points[:, 1] > bboxes[:, 1]) & \
               (points[:, 1] < bboxes[:, 3])


def bboxes_area(bboxes):
    """Compute the area of an array of boxes."""
    w = (bboxes[:, 2] - bboxes[:, 0] + 1)
    h = (bboxes[:, 3] - bboxes[:, 1] + 1)
    areas = w * h

    return areas
