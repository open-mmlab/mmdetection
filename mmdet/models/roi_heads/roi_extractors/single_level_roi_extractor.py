# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .base_roi_extractor import BaseRoIExtractor


@MODELS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """从单层特征图上提取ROI特征.

    如果有多个层级特征图，则每个ROI根据其大小映射到其中之一. 映射规则参见
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): 指定 RoI 层类型和参数.
        out_channels (int): RoI 层的输出维度.
        featmap_strides (List[int]): 输入特征图对应的stride.
        finest_scale (int): 映射到最大特征图上的尺寸阈值.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): 初始化配置字典.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 finest_scale: int = 56,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """按比例将 rois 映射到相应的特征图上.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: 每个 RoI 的特征层索引(0-base), [k, ]
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None):
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
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)
        # [k,] 其中每个值代表每个roi所属特征图ind
        # k为一个batch中所有层级上的roi的总数,由于经过rcnn中的sampler处理
        # 所以其被限制最大为512*batch.
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:  # 对roi宽高进行缩放
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)  # 当前特征图中所有的roi索引
            if inds.numel() > 0:
                rois_ = rois[inds]  # 当前特征图中所有的roi
                # 指定的roi层对指定的roi在其所属特征图上进行RoIPool/RoIAlign
                # self.roi_layers[i] RoIAlign(output_size=(7, 7), spatial_scale=1/(4/8/16/32),
                # sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
                # feats[i] -> [bs, self.out_channels, f_h, f_w]
                # rois_ -> [1935, 5] 其中1935不具备普适性
                # roi_feats_t -> [1935, self.out_channels, 7, 7]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t  # 逐层填充输出结果
            else:
                # 有时一些层级上的特征图没有匹配到任何roi,这会导致一个GPU中的计算图不完整,
                # 并与其他GPU中的计算图不同,会导致挂起错误
                # 因此,我们添加以下部分代码以确保每个层级特征图都包含在计算图中以避免运行时错误
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
