# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class TransformerDetector(BaseDetector):
    """Base class for Transformer-based detectors.

    Transformer-based detectors use an encoder to process output features of
    backbone/neck, then several queries interactive with encoded features and
    do the regression and classification.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder_cfg: OptConfigType = None,
                 decoder_cfg: OptConfigType = None,
                 positional_encoding_cfg: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 num_query: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.positional_encoding_cfg = positional_encoding_cfg
        self.num_query = num_query

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    def _init_layers(self) -> None:
        self._init_transformer()

    def _init_transformer(self) -> None:
        """1. Initialize positional_encoding
           2. Initialize encoder and decoder of transformer
           3. Get self.embed_dims from the transformer
           4. Initialize query_embed"""
        raise NotImplementedError(
            'The _init_transformer should be implemented for the detector.')

    # def init_weight  # TODO !!!!

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec = self.forward_transformer(**transformer_inputs_dict)
        losses = self.bbox_head.loss(outs_dec, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec = self.forward_transformer(**transformer_inputs_dict)
        results_list = self.bbox_head.predict(
            outs_dec, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec = self.forward_transformer(**transformer_inputs_dict)
        results = self.bbox_head.forward(outs_dec)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_pretransformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Dict[str, Tensor]:
        """1. Get batch padding mask.
           2. Convert image feature maps to sequential features.
           3. Get image positional embedding of features.
           4. Prepare decoder queries."""
        raise NotImplementedError(
            'The forward_pretransformer should be implemented '
            'for the detector.')

    def forward_transformer(self, **kwargs) -> Tuple[Tensor]:
        """Process sequential features with transformer."""
        raise NotImplementedError(
            'The forward_transformer should be implemented for the detector.')
