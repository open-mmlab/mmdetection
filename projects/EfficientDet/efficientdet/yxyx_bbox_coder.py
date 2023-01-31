# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes, get_box_tensor


@TASK_UTILS.register_module()
class YXYXDeltaXYWHBBoxCoder(DeltaXYWHBBoxCoder):

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Source boxes,
                e.g., object proposals.
            gt_bboxes (torch.Tensor or :obj:`BaseBoxes`): Target of the
                transformation, e.g., ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = YXbbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            Union[torch.Tensor, :obj:`BaseBoxes`]: Decoded boxes.
        """
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            # single image decode
            decoded_bboxes = YXdelta2bbox(bboxes, pred_bboxes, self.means,
                                          self.stds, max_shape, wh_ratio_clip,
                                          self.clip_border, self.add_ctr_clamp,
                                          self.ctr_clamp)
        else:
            if pred_bboxes.ndim == 3 and not torch.onnx.is_in_onnx_export():
                warnings.warn(
                    'DeprecationWarning: onnx_delta2bbox is deprecated '
                    'in the case of batch decoding and non-ONNX, '
                    'please use “delta2bbox” instead. In order to improve '
                    'the decoding speed, the batch function will no '
                    'longer be supported. ')
            decoded_bboxes = YXonnx_delta2bbox(bboxes, pred_bboxes, self.means,
                                               self.stds, max_shape,
                                               wh_ratio_clip, self.clip_border,
                                               self.add_ctr_clamp,
                                               self.ctr_clamp)

        if self.use_box_type:
            assert decoded_bboxes.size(-1) == 4, \
                ('Cannot warp decoded boxes with box type when decoded boxes'
                 'have shape of (N, num_classes * 4)')
            decoded_bboxes = HorizontalBoxes(decoded_bboxes)
        return decoded_bboxes


def YXdelta2bbox(rois,
                 deltas,
                 means=(0., 0., 0., 0.),
                 stds=(1., 1., 1., 1.),
                 max_shape=None,
                 hw_ratio_clip=1000 / 16,
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

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
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dyx = denorm_deltas[:, :2]
    dhw = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pyx = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    phw = (rois_[:, 2:] - rois_[:, :2])

    dyx_hw = phw * dyx

    max_ratio = np.abs(np.log(hw_ratio_clip))
    if add_ctr_clamp:
        dyx_hw = torch.clamp(dyx_hw, max=ctr_clamp, min=-ctr_clamp)
        dhw = torch.clamp(dhw, max=max_ratio)
    else:
        dhw = dhw.clamp(min=-max_ratio, max=max_ratio)

    gyx = pyx + dyx_hw
    ghw = phw * dhw.exp()
    y1x1 = gyx - (ghw * 0.5)
    y2x2 = gyx + (ghw * 0.5)
    ymin, xmin = y1x1[:, 0].reshape(-1, 1), y1x1[:, 1].reshape(-1, 1)
    ymax, xmax = y2x2[:, 0].reshape(-1, 1), y2x2[:, 1].reshape(-1, 1)
    bboxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


def YXbbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

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
    py = (proposals[..., 0] + proposals[..., 2]) * 0.5
    px = (proposals[..., 1] + proposals[..., 3]) * 0.5
    ph = proposals[..., 2] - proposals[..., 0]
    pw = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dy, dx, dh, dw], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def YXonnx_delta2bbox(rois,
                      deltas,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.),
                      max_shape=None,
                      wh_ratio_clip=16 / 1000,
                      clip_border=True,
                      add_ctr_clamp=False,
                      ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B. Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
            Default 16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4), where 4 represent
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
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = deltas.new_tensor(means).view(1,
                                          -1).repeat(1,
                                                     deltas.size(-1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
    denorm_deltas = deltas * stds + means
    dy = denorm_deltas[..., 0::4]
    dx = denorm_deltas[..., 1::4]
    dh = denorm_deltas[..., 2::4]
    dw = denorm_deltas[..., 3::4]

    y1, x1 = rois[..., 0], rois[..., 1]
    y2, x2 = rois[..., 2], rois[..., 3]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx)
    py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
    # Compute width/height of each roi
    pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
    ph = (y2 - y1).unsqueeze(-1).expand_as(dh)

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())

    if clip_border and max_shape is not None:
        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat(
            [max_shape] * (deltas.size(-1) // 2),
            dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes
