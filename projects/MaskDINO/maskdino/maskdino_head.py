# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
from .maskdino_encoder_layers import MaskDINOEncoder
from .maskdino_decoder_layers import MaskDINODecoder
# from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
# from ..utils import get_uncertain_point_coords_with_randomness
# from .anchor_free_head import AnchorFreeHead
# from .maskformer_head import MaskFormerHead


@MODELS.register_module()
class MaskDINOHead(nn.Module):

    def __init__(
        self,
        # input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        encoder: OptConfigType,
        decoder: OptConfigType,
        loss_weight: float = 1.0,
        ignore_value: int = -1
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # self.in_features = [k for k, v in input_shape]  # useless
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = MaskDINOEncoder(**encoder)
        self.predictor = MaskDINODecoder(**decoder)

        self.num_classes = num_classes

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets=targets)

    def layers(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(features, mask)

        predictions = self.predictor(multi_scale_features, mask_features,
                                     mask, targets=targets)
        return predictions

    def loss(self, feats, batch_data_samples):
        pass

    def predict(self, feats, batch_data_samples):
        pass
