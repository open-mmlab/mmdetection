# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes, get_box_tensor
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class DeltaXYWHBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    按照 `R-CNN <https://arxiv.org/abs/1311.2524>` 中的做法,
    下列代码可计算出基础box与目标box之间的修正系数,也可根据基础box与修正系数计算出目标box.

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): 是否限制box范围在图像内部. 默认True.
        add_ctr_clamp (bool): 是否进行中心限制, 当进行该操作时, 预测框被固定住是因为
            它的中心离anchor的中心太远. 仅在 YOLOF 中使用. 默认 False.
        ctr_clamp (int): 限制最大像素偏移. 仅在 YOLOF 中使用. 默认 32.
    """

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1.),
                 clip_border: bool = True,
                 add_ctr_clamp: bool = False,
                 ctr_clamp: int = 32,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

<<<<<<< HEAD:mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
    def encode(self, bboxes, gt_bboxes):
        """计算box修正到gt box的修正系数.但计算得到后需要再次进行归一化,
        进行该操作是由于回归的loss通常要比分类loss小的多.为了提高多任务学习的有效性,
        也即平衡多个loss.需要对回归值进行归一化,这普遍应用于rcnn系列网络.
        参见于Cascade R-CNN论文3.1节内容.

        Args:
            bboxes (torch.Tensor): 初始box.
            gt_bboxes (torch.Tensor): gt box.
=======
    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_bboxes: Union[Tensor, BaseBoxes]) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Source boxes,
                e.g., object proposals.
            gt_bboxes (torch.Tensor or :obj:`BaseBoxes`): Target of the
                transformation, e.g., ground-truth boxes.
>>>>>>> mmdetection/main:mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py

        Returns:
            torch.Tensor: 修正系数
        """
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

<<<<<<< HEAD:mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """在box上应用修正系数以得到修正后的box.

        Args:
            bboxes (torch.Tensor): 初始box. [bs, N, 4] or [N, 4], N = na * H * W.
            pred_bboxes (Tensor): box的修正系数.
               [bs, N, nc * 4] or [bs, N, 4] or [N, nc * 4] or [N, 4].
=======
    def decode(
        self,
        bboxes: Union[Tensor, BaseBoxes],
        pred_bboxes: Tensor,
        max_shape: Optional[Union[Sequence[int], Tensor,
                                  Sequence[Sequence[int]]]] = None,
        wh_ratio_clip: Optional[float] = 16 / 1000
    ) -> Union[Tensor, BaseBoxes]:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
>>>>>>> mmdetection/main:mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): box的最大边界, shape可能为
               [H, W, C] or [H, W]. 如果box的shape为 [bs, N, 4], 那么max_shape就应该是
               [Sequence[int], ] * bs,最外层格式也可能是元组
            wh_ratio_clip (float, optional): 允许的宽高比.

        Returns:
<<<<<<< HEAD:mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
            torch.Tensor: 修正后的box.
=======
            Union[torch.Tensor, :obj:`BaseBoxes`]: Decoded boxes.
>>>>>>> mmdetection/main:mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
        """
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            # 在单张图像上对box修正
            decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means,
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
            decoded_bboxes = onnx_delta2bbox(bboxes, pred_bboxes, self.means,
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


<<<<<<< HEAD:mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
@mmcv.jit(coderize=True)
def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """计算基础box与目标box之间的修正系数.
=======
def bbox2delta(
    proposals: Tensor,
    gt: Tensor,
    means: Sequence[float] = (0., 0., 0., 0.),
    stds: Sequence[float] = (1., 1., 1., 1.)
) -> Tensor:
    """Compute deltas of proposals w.r.t. gt.
>>>>>>> mmdetection/main:mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py

    通常来说是计算x, y, w, h四个维度的修正系数

    Args:
        proposals (Tensor): 基础box, shape (N, ..., 4)
        gt (Tensor): 目标box, shape (N, ..., 4)
        means (Sequence[float]): 计算出修正系数后要减去的均值
        stds (Sequence[float]): 计算出修正系数后要除以的方差

    Returns:
        Tensor: 计算得到的修正系数,shape与输入box一致
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


<<<<<<< HEAD:mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """利用修正系数修正基础box.基础box可能是anchor也可能是roi
=======
def delta2bbox(rois: Tensor,
               deltas: Tensor,
               means: Sequence[float] = (0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1.),
               max_shape: Optional[Union[Sequence[int], Tensor,
                                         Sequence[Sequence[int]]]] = None,
               wh_ratio_clip: float = 16 / 1000,
               clip_border: bool = True,
               add_ctr_clamp: bool = False,
               ctr_clamp: int = 32) -> Tensor:
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.
>>>>>>> mmdetection/main:mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py

    Args:
        rois (Tensor): 基础box. [N, 4].
        deltas (Tensor): 修正系数. [N, nc * 4] or [N, 4]. N = na * W * H
        means (Sequence[float]): 逆归一化的均值. 默认 [0., 0., 0., 0.].
        stds (Sequence[float]): 逆归一化的方差. 默认 [1., 1., 1., 1.].
        max_shape (tuple[int, int]): 修正后的box最大边界, (H, W). 默认 None.
        wh_ratio_clip (float): 修正后的box最大宽高比. 默认 16 / 1000.
        clip_border (bool, optional): 是否截断box超出图像边界外的区域. 默认 True.
        add_ctr_clamp (bool): 是否增加一个偏移限制. 当设置为True时,偏移距离会被约束,
            以避免离anchor的中心太远. 仅用于YOLOF. 默认 False.
        ctr_clamp (int): 中心坐标的最大偏移像素. 仅用于YOLOF. 默认 32.

    Returns:
        Tensor: 修正后的box [N, num_classes * 4] or [N, 4], x1, y1, x2, y2.

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

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


def onnx_delta2bbox(rois: Tensor,
                    deltas: Tensor,
                    means: Sequence[float] = (0., 0., 0., 0.),
                    stds: Sequence[float] = (1., 1., 1., 1.),
                    max_shape: Optional[Union[Sequence[int], Tensor,
                                              Sequence[Sequence[int]]]] = None,
                    wh_ratio_clip: float = 16 / 1000,
                    clip_border: Optional[bool] = True,
                    add_ctr_clamp: bool = False,
                    ctr_clamp: int = 32) -> Tensor:
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
    dx = denorm_deltas[..., 0::4]
    dy = denorm_deltas[..., 1::4]
    dw = denorm_deltas[..., 2::4]
    dh = denorm_deltas[..., 3::4]

    x1, y1 = rois[..., 0], rois[..., 1]
    x2, y2 = rois[..., 2], rois[..., 3]
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
