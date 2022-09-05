# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois, num_levels):
        """按比例将 rois 映射到相应的特征图上.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): 特征图层数.

        Returns:
            Tensor: 每个 RoI 的特征层索引(从 0 开始), shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # 兼容mask-rcnn 导出到 onnx时遇到的问题,
            # 先复制出一个基础的[k, 1]的tensor,再在第二维扩充到指定长度再reshape最后乘零
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(  # 首先初始化所有roi的输出结果
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)
        # 此处以faster_rcnn_r50_fpn.py,单卡训练过程为例
        # target_lvls.shape -> [k,], 其中每个值代表每个roi所属特征图ind
        # k为一个batch中所有层级上的roi的总数,由于经过rcnn中的sampler处理
        # 所以其被限制最大为512*batch,一般情况下,k会随着训练时间增加而降低
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:  # 对roi宽高进行缩放
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i  # [k,]
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i = rois_i * mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t = roi_feats_t * mask_exp
                roi_feats = roi_feats + roi_feats_t
                continue
            # [k, 1] -> [k, 1]
            inds = mask.nonzero(as_tuple=False).squeeze(1)  # 当前特征图中所有的roi索引
            if inds.numel() > 0:
                rois_ = rois[inds]  # 当前特征图中所有的roi
                # 指定的roi层对指定的roi在其所属特征图上进行RoIPool/RoIAlign
                # self.roi_layers[i] RoIAlign(output_size=(7, 7), spatial_scale=1/(4/8/16/32),
                # sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
                # feats[i].shape -> torch.Size([bs, self.out_channels, f_h, f_w])
                # rois_ -> torch.Size([1935, 5]) 其中1935不具备普适性
                # roi_feats_t.shape -> torch.Size([1935, self.out_channels, 7, 7])
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t  # 逐层填充输出结果
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats = roi_feats + sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
