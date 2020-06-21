"""Generic RoI Extractor.

A novel Region of Interest Extraction Layer for Instance Segmentation.
"""

from torch import nn

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.ops.plugin import build_plugin_layer
from .single_level import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class SumGenericRoiExtractor(SingleRoIExtractor):
    """Extract RoI features from all summed feature maps levels.

    https://arxiv.org/abs/2004.13665

    Args:
        pre_cfg (dict): Specify pre-processing modules.
        post_cfg (dict): Specify post-processing modules.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`SingleRoIExtractor`.
    """

    def __init__(self, pre_cfg, post_cfg, **kwargs):
        super(SumGenericRoiExtractor, self).__init__(**kwargs)

        # build pre/post processing modules
        self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]
        self.relu = nn.ReLU(inplace=False)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            # apply pre-processing to a RoI extracted from each layer
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats_t = self.pre_module(roi_feats_t)
            roi_feats_t = self.relu(roi_feats_t)
            # and sum them all
            roi_feats += roi_feats_t

        # apply post-processing before return the result
        x = self.post_module(roi_feats)
        return x
