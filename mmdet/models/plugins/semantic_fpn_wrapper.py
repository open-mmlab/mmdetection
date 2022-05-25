# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule
from mmcv.runner import ModuleList

from ..builder import PLUGIN_LAYERS
from ..utils import ConvUpsample

@PLUGIN_LAYERS.register_module()
class SemanticFPNWrapper(BaseModule):
    """Implementation of Semantic FPN used in Panoptic FPN.

    Args:
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        out_channels (int): Number of channels in output features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        cat_coors_level (int): Indicate which level will add
            position embdding.
        upsample_times (int): Upsample time of end level.
        num_aux_convs (int): number of aux conv for semantic
            segmentation.
        out_act_cfg (dict): Config dict for output 
            activation layer. Default: ReLU.
        conv_cfg (dict): Dictionary to construct and config
            conv layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Use ``GN`` by default.
    """

    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 start_level,
                 end_level,
                 positional_encoding=None,
                 cat_coors_level=-1,
                 upsample_times=2,
                 num_aux_convs=0,
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticFPNWrapper, self).__init__()

        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cat_coors_level = cat_coors_level
        self.upsample_times = upsample_times
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
        else:
            self.positional_encoding = None

        self.conv_upsample_layers = ModuleList()
        for i in range(start_level, end_level):
            num_layers = i-(end_level-upsample_times-1)
            self.conv_upsample_layers.append(
                ConvUpsample(
                    in_channels,
                    inner_channels,
                    num_layers= num_layers+1 if num_layers > 0 else 1,
                    num_upsample= num_layers if num_layers > 0 else 0,
                    stride = 1 if num_layers >= 0 else 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        in_channels = self.inner_channels

        self.conv_pred = ConvModule(
            in_channels,
            self.out_channels,
            1,
            padding=0,
            conv_cfg=self.conv_cfg,
            act_cfg=out_act_cfg,
            norm_cfg=self.norm_cfg)

        self.num_aux_convs = num_aux_convs # equal to 1
        self.aux_convs = nn.ModuleList()
        for i in range(num_aux_convs):
            self.aux_convs.append(
                ConvModule(
                    in_channels,
                    self.out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=out_act_cfg,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def generate_coord(self, input_feat):
        x_range = torch.linspace(
            -1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(
            -1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, inputs):
        """
        Args:
            inputs (List[Tensor]):feature of each feature map, \
                each has shape (N,C,H_i,W_i),
        return:
            List[Tensor]: loc_feats: (N,C,H,W) 8x downsample
                          sem_feature: (N,C,H,W) 8x downsample
        """
        mlvl_feats = []
        for i in range(self.start_level, self.end_level):
            input_p = inputs[i]
            if i == self.cat_coors_level:
                if self.positional_encoding is not None:
                    ignore_mask = input_p.new_zeros(
                        (input_p.shape[0], input_p.shape[-2],
                         input_p.shape[-1]),
                        dtype=torch.bool)
                    positional_encoding = self.positional_encoding(ignore_mask)
                    input_p = input_p + positional_encoding.contiguous()

            mlvl_feats.append(self.conv_upsample_layers[i](input_p))

        out_features = sum(mlvl_feats)
        out = self.conv_pred(out_features)
        outs = [out]
        if self.num_aux_convs > 0:
            for conv in self.aux_convs:
                outs.append(conv(out_features))
        return outs
