# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from mmcv.ops import (ContextBlock, Conv2d, ConvTranspose2d, ConvWS2d,
                      CornerPool, DeformConv, DeformConvPack, DeformRoIPooling,
                      DeformRoIPoolingPack, GeneralizedAttention, Linear,
                      MaskedConv2d, MaxPool2d, ModulatedDeformConv,
                      ModulatedDeformConvPack, ModulatedDeformRoIPoolingPack,
                      NonLocal2D, RoIAlign, RoIPool, SAConv2d,
                      SigmoidFocalLoss, SimpleRoIAlign, batched_nms,
                      build_plugin_layer, conv_ws_2d, deform_conv,
                      deform_roi_pooling, get_compiler_version,
                      get_compiling_cuda_version, modulated_deform_conv, nms,
                      nms_match, point_sample, rel_roi_point_to_rel_img_point,
                      roi_align, roi_pool, sigmoid_focal_loss, soft_nms)

# yapf: enable

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'DeformConv', 'DeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pooling', 'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'MaskedConv2d', 'ContextBlock', 'GeneralizedAttention', 'NonLocal2D',
    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
    'conv_ws_2d', 'build_plugin_layer', 'batched_nms', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d'
]
