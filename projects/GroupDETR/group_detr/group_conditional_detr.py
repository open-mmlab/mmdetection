# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.conditional_detr import ConditionalDETR
from mmdet.models.layers.positional_encoding import SinePositionalEncoding
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from .group_conditional_detr_decoder import \
    GroupConditionalDetrTransformerDecoder


@MODELS.register_module()
class GroupConditionalDETR(ConditionalDETR):
    r"""Implementation of `Fast DETR Training with Group-Wise
     One-to-Many Assignment.

    <https://arxiv.org/abs/2207.13085>`_.

    Code is modified from the `official github repo
    <https://github.com/Atten4Vis/ConditionalDETR/tree/GroupDETR>`_.

    Args:
        num_query_groups (int): The number of decoder query groups.
    """

    def __init__(self, *arg, num_query_groups: int = 1, **kwargs) -> None:
        self.num_query_groups = num_query_groups
        super().__init__(*arg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = GroupConditionalDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(
            self.num_queries * self.num_query_groups, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        if self.training:
            # use all groups in training
            query_pos = self.query_embedding.weight
        else:
            # only use one group in inference
            query_pos = self.query_embedding.weight[:self.num_queries]
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
