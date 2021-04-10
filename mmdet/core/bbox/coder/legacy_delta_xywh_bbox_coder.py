import mmcv
import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class LegacyDeltaXYWHBBoxCoder(BaseBBoxCoder):
    """Legacy Delta XYWH BBox coder used in MMDet V1.x.

    Following the practice in R-CNN [1]_, this coder encodes bbox (x1, y1, x2,
    y2) into delta (dx, dy, dw, dh) and decodes delta (dx, dy, dw, dh)
    back to original bbox (x1, y1, x2, y2).

    Note:
        The main difference between :class`LegacyDeltaXYWHBBoxCoder` and
        :class:`DeltaXYWHBBoxCoder` is whether ``+ 1`` is used during width and
        height calculation. We suggest to only use this coder when testing with
        MMDet V1.x models.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Args:
        target_means (Sequence[float]): denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = legacy_bbox2delta(bboxes, gt_bboxes, self.means,
                                           self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = legacy_delta2bbox(bboxes, pred_bboxes, self.means,
                                           self.stds, max_shape, wh_ratio_clip)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def legacy_bbox2delta(proposals,
                      gt,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt in the MMDet V1.x manner.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of `delta2bbox()`

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.jit(coderize=True)
def legacy_delta2bbox(rois,
                      deltas,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.),
                      max_shape=None,
                      wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes in the MMDet V1.x manner.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of `bbox2delta()`

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> legacy_delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.5000, 1.5000],
                [0.0000, 0.0000, 5.2183, 5.2183],
                [0.0000, 0.1321, 7.8891, 0.8679],
                [5.3967, 2.4251, 6.0033, 3.7749]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right

    # The true legacy box coder should +- 0.5 here.
    # However, current implementation improves the performance when testing
    # the models trained in MMDetection 1.X (~0.5 bbox AP, 0.2 mask AP)
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes
