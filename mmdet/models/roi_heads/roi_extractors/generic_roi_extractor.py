# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

from mmcv.cnn.bricks import build_plugin_layer
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from .base_roi_extractor import BaseRoIExtractor


@MODELS.register_module()
class GenericRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Defaults to 'sum'.
        pre_cfg (:obj:`ConfigDict` or dict): Specify pre-processing modules.
            Defaults to None.
        post_cfg (:obj:`ConfigDict` or dict): Specify post-processing modules.
            Defaults to None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation: str = 'sum',
                 pre_cfg: OptConfigType = None,
                 post_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

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
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if num_levels == 1:
            return self.roi_layers[0](feats[0], rois)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode
        start_channels = 0
        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            end_channels = start_channels + roi_feats_t.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t = self.pre_module(roi_feats_t)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats += roi_feats_t
            else:
                # and concat them along channel dimension
                roi_feats[:, start_channels:end_channels] = roi_feats_t
            # update channels starting position
            start_channels = end_channels
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels

        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats
