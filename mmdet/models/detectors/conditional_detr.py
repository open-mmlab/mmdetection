# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from ..layers import (ConditionalDetrTransformerDecoder,
                      DetrTransformerEncoder, SinePositionalEncoding)
from .detr import DETR


@MODELS.register_module()
class ConditionalDETR(DETR):
    r"""Implementation of `Conditional DETR for Fast Training Convergence.

    <https://arxiv.org/abs/2108.06152>`_.

    Code is modified from the `official github repo
    <https://github.com/Atten4Vis/ConditionalDETR>`_.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references` of the decoder output.

            - hidden_states (Tensor): Has shape
                (num_decoder_layers, bs, num_queries, dim)
            - references (Tensor): Has shape
                (bs, num_queries, 2)
        """

        hidden_states, references = self.decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references)
        return head_inputs_dict
