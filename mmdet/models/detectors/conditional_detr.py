# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple
from torch import Tensor
from .detr import DETR

import torch
import torch.nn as nn

from ..layers import (DetrTransformerEncoder, ConditionalDetrTransformerDecoder,
                      SinePositionalEncoding)

from mmdet.registry import MODELS


@MODELS.register_module()
class ConditionalDETR(DETR):
    r"""Implementation of `Conditional DETR for Fast Training Convergence.

    <https://arxiv.org/abs/2108.06152>`_.

    Code is modified from the `official github repo
    <https://github.com/Atten4Vis/ConditionalDETR>`_.
    """
    def __init__(self, *arg, group_detr=1, **kwargs) -> None:
        self.group_detr = group_detr
        super().__init__(*arg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_query * self.group_detr, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
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
                has shape (num_feat, bs, dim).

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

        batch_size = memory.size(1)
        if self.training:
            query_pos = self.query_embedding.weight
        else:
            query_pos = self.query_embedding.weight[:self.num_query]
        # (num_query, dim) -> (num_query, bs, dim)
        query_pos = query_pos.unsqueeze(1).repeat(1, batch_size, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        query_pos: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (num_query, bs, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (num_query, bs, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (num_feat, bs, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (num_feat, bs, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.#TODO
        """
        # (num_decoder_layers, num_query, bs, dim)
        hidden_states, reference_points = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        hidden_states = hidden_states.transpose(1, 2)
        head_inputs_dict = dict(hidden_states=hidden_states, reference_points=reference_points)
        return head_inputs_dict


