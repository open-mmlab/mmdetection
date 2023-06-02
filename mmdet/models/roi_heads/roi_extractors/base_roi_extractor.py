# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv import ops
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import ConfigType, OptMultiConfig


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list[int]): Strides of input feature maps.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
<<<<<<< HEAD
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 init_cfg=None):
        super(BaseRoIExtractor, self).__init__(init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)  # 传入roi配置以构建ROI层
=======
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
>>>>>>> mmdetection/main
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
<<<<<<< HEAD
    def num_inputs(self):
        """int: 输入特征图的数量."""
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """构建 RoI 算子从每个级别的特征图中提取特征.

        Args:
            layer_cfg (dict): 构造 RoI 层的配置字典. 选项是 "mmcv.ops" 下的模块,例如“RoIAlign”
                Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
=======
    def num_inputs(self) -> int:
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg: ConfigType,
                         featmap_strides: List[int]) -> nn.ModuleList:
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
                config RoI layer operation. Options are modules under
                ``mmcv/ops`` such as ``RoIAlign``.
            featmap_strides (list[int]): The stride of input feature map w.r.t
>>>>>>> mmdetection/main
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
<<<<<<< HEAD
            nn.ModuleList: 多层级上的RoI特征提取模块.
=======
            :obj:`nn.ModuleList`: The RoI extractor modules for each level
                feature map.
>>>>>>> mmdetection/main
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(  # 构建多个不同层级的ROI层,
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

<<<<<<< HEAD
    def roi_rescale(self, rois, scale_factor):
        """按照scale_factor缩放RoI的宽高,但由于RoI是x1,y1,x2,y2表示,
            所以需要转换为x,y,w,h格式对wh进行缩放再转为x1,y1,x2,y2表示.
=======
    def roi_rescale(self, rois: Tensor, scale_factor: float) -> Tensor:
        """Scale RoI coordinates by scale factor.
>>>>>>> mmdetection/main

        Args:
            rois (Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @abstractmethod
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        pass
