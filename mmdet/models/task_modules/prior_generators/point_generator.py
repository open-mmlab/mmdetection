# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.registry import TASK_UTILS

DeviceType = Union[str, torch.device]


@TASK_UTILS.register_module()
class PointGenerator:

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self,
                    featmap_size: Tuple[int, int],
                    stride=16,
                    device: DeviceType = 'cuda') -> Tensor:
        """Generate grid points of a single level.

        Args:
            featmap_size (tuple[int, int]): Size of the feature maps.
            stride (int): The stride of corresponding feature map.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: grid point in a feature map.
        """
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self,
                    featmap_size: Tuple[int, int],
                    valid_size: Tuple[int, int],
                    device: DeviceType = 'cuda') -> Tensor:
        """Generate valid flags of anchors in a feature map.

        Args:
            featmap_sizes (list(tuple[int, int])): List of feature map sizes in
                multiple feature levels.
            valid_shape (tuple[int, int]): The valid shape of the image.
            device (str | torch.device): Device where the anchors will be
                put on.

        Return:
            torch.Tensor: Valid flags of anchors in a level.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid


@TASK_UTILS.register_module()
class MlvlPointGenerator:
    """基于 2D points-based检测器中多层级特征图的标准点生成器.

    Args:
        strides (list[int] | list[tuple[int, int]]): 多层级特征图上anchor下采样倍数,
            如果参数格式为后者, 那么按(w, h)顺序排列.
        offset (float): 点的偏移量, 该值用相应的stride进行归一化. Defaults to 0.5.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 offset: float = 0.5) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
<<<<<<< HEAD:mmdet/core/anchor/point_generator.py
    def num_base_priors(self):
        """list[int]: 各层级上特征点的先验(点)数,它在anchor_base中称为先验(框)数"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x, indexing='ij')
=======
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
>>>>>>> mmdetection/main:mmdet/models/task_modules/prior_generators/point_generator.py
        if row_major:
            # 警告 .flatten() 会导致 ONNX 导出错误,因此必须在此处使用 reshape
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
<<<<<<< HEAD:mmdet/core/anchor/point_generator.py
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):
        """生成多层级的先验点.

        Args:
            featmap_sizes (list[tuple]): 多层级的特征图大小, [[h, w],] * num_level.
            dtype (:obj:`dtype`): priors的类型. Default: torch.float32.
            device (str): 在该设备上生成priors.
            with_stride (bool): 是否将stride合并到先验点的坐标上去.
=======
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device: DeviceType = 'cuda',
                    with_stride: bool = False) -> List[Tensor]:
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device where the anchors will be
                put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.
>>>>>>> mmdetection/main:mmdet/models/task_modules/prior_generators/point_generator.py

        Return:
            list[torch.Tensor]: 多个特征图上的先验点. 默认:[[h*w, 2],] * num_level
            当with_stride为 False 时, [h*w, 2], 2 -> [x, y]
            当with_stride为 True 时, [h*w, 4], 4 -> [x, y, stride_w, stride_h]
                priors中心偏移grid左上角offset*(stride_h/w)/2个单位像素.
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
<<<<<<< HEAD:mmdet/core/anchor/point_generator.py
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """生成单层特征图上的网格点.
=======
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: DeviceType = 'cuda',
                                 with_stride: bool = False) -> Tensor:
        """Generate grid Points of a single level.
>>>>>>> mmdetection/main:mmdet/models/task_modules/prior_generators/point_generator.py

        注意:
            此函数通常由方法 ``self.grid_priors`` 调用.

        Args:
<<<<<<< HEAD:mmdet/core/anchor/point_generator.py
            featmap_size (tuple[int]): 特征图的大小, 表示为[h, w].
            level_idx (int): 对应特征图的索引.
            dtype (:obj:`dtype`): priors的类型. 默认: torch.float32.
            device (str, optional): 在该设备上生成priors.
            with_stride (bool): 是否将stride与prior中心坐标合并到一起.
=======
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.
>>>>>>> mmdetection/main:mmdet/models/task_modules/prior_generators/point_generator.py

        Return:
            Tensor: 单个特征图上的先验点,偏移grid左上角offset*(stride_h/w)/2个单位像素.
            当with_stride为 False 时, Tensor的形状应为 (h*w, 2), 2 代表 (x, y),
            当with_stride为 True 时, Tensor的形状应为 (h*w, 4),
            4 代表 (x, y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # 让shift_x保持为Tensor 而非int, 是为了我们可以正确的转换到ONNX
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # 同上
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # 进行 ONNX 导出时,使用 shape[0] 而不是 len(shift_xx)
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple[int],
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                arrange as (h, w).
            device (str | torch.device): The device where the anchors will be
                put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 device: DeviceType = 'cuda') -> Tensor:
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str | torch.device): The device where the flags will be
            put on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs: Tensor,
                      featmap_size: Tuple[int],
                      level_idx: int,
                      dtype: torch.dtype = torch.float32,
                      device: DeviceType = 'cuda') -> Tensor:
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris
