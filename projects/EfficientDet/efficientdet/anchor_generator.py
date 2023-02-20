# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from mmdet.models.task_modules.prior_generators.anchor_generator import \
    AnchorGenerator
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes

DeviceType = Union[str, torch.device]


@TASK_UTILS.register_module()
class YXYXAnchorGenerator(AnchorGenerator):

    def gen_single_level_base_anchors(self,
                                      base_size: Union[int, float],
                                      scales: Tensor,
                                      ratios: Tensor,
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """

        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            y_center - 0.5 * hs,
            x_center - 0.5 * ws,
            y_center + 0.5 * hs,
            x_center + 0.5 * ws,
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: DeviceType = 'cuda') -> Tensor:
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int, int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_yy, shift_xx, shift_yy, shift_xx], dim=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        if self.use_box_type:
            all_anchors = HorizontalBoxes(all_anchors)

        return all_anchors
