from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList
from torch import Tensor
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from mmdet.models.layers.transformer.deformable_detr_layers import \
    DeformableDetrTransformerEncoderLayer
from mmdet.utils import ConfigType, OptConfigType
from .modeling_bert import BertAttention, BertIntermediate, BertOutput


class BiMultiHeadAttention(nn.Module):

    def __init__(self,
                 v_dim,
                 l_dim,
                 embed_dim,
                 num_heads,
                 dropout=0.1,
                 stable_softmax_2d=False,
                 clamp_min_for_underflow=True,
                 clamp_max_for_overflow=True):

        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (self.head_dim * self.num_heads == self.embed_dim
                ), f'embed_dim must be divisible by num_heads \
            (got `embed_dim`: {self.embed_dim} and `num_heads`: \
                {self.num_heads}).'

        self.scale = self.head_dim**(-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = stable_softmax_2d
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, lang, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim
                      )  # (bs * 8, -1, embed_dim//8)
        query_states = self._shape(query_states, tgt_len, bsz).view(
            *proj_shape)  # (bs * 8, seq_len_img, embed_dim//8)
        key_states = key_states.view(
            *proj_shape)  # (bs * 8, seq_len_text, embed_dim//8)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(
            1, 2))  # (bs * 8, seq_len_img, seq_len_text)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size \
                    {(bsz * self.num_heads, tgt_len, src_len)}, \
                        but is {attn_weights.size()}')

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (
            attn_weights_T -
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)
        # assert attention_mask_l.dtype == torch.int64
        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)  # (bs, seq_len)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(
                1)  # (bs, 1, 1, seq_len)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size \
                          {(bsz, 1, tgt_len, src_len)}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(
            attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(
            attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len,
                                    self.head_dim):
            raise ValueError(f'`attn_output_v` should be of size \
                    {(bsz, self.num_heads, tgt_len, self.head_dim)}, \
                        but is {attn_output_v.size()}')

        if attn_output_l.size() != (bsz * self.num_heads, src_len,
                                    self.head_dim):
            raise ValueError(f'`attn_output_l` should be of size \
                {(bsz, self.num_heads, src_len, self.head_dim)}, \
                but is {attn_output_l.size()}')

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len,
                                           self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len,
                                           self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class BiAttentionBlockForCheckpoint(nn.Module):

    def __init__(self,
                 v_dim,
                 l_dim,
                 embed_dim,
                 num_heads,
                 dropout=0.1,
                 drop_path=.0,
                 init_values=1e-4,
                 stable_softmax_2d=False,
                 clamp_min_for_underflow=True,
                 clamp_max_for_overflow=True):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the
                        Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            stable_softmax_2d=stable_softmax_2d,
            clamp_min_for_underflow=clamp_min_for_underflow,
            clamp_max_for_overflow=clamp_max_for_overflow)

        # add layer scale for training stability
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(
            init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(
            init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, vision, lang, attention_mask_l=None, task=None):
        # v: visual features, (bs, sigma(HW), 256)
        # l: language features, (bs, seq_len, 768)
        vision = self.layer_norm_v(vision)
        lang = self.layer_norm_l(lang)
        delta_v, delta_l = self.attn(
            vision, lang, attention_mask_l=attention_mask_l)
        vision = vision + self.drop_path(self.gamma_v * delta_v)
        lang = lang + self.drop_path(self.gamma_l * delta_l)
        return vision, lang


class BertEncoderLayer(BertPreTrainedModel):

    def __init__(self,
                 config,
                 clamp_min_for_underflow=False,
                 clamp_max_for_overflow=False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = BertAttention(config, clamp_min_for_underflow,
                                       clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs['lang']
        hidden_states = language_dict_features['hidden']
        attention_mask = language_dict_features['masks']

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs
        hidden_states = outputs[0]

        language_dict_features['hidden'] = hidden_states

        features_dict = {
            'visual': inputs['visual'],
            'lang': language_dict_features
        }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VLFuse(BaseModule):
    """Early Fusion Module."""

    def __init__(self,
                 lang_model: str = 'bert-base-uncased',
                 lang_dim: int = 768,
                 stable_softmax_2d: bool = False,
                 clamp_min_for_underflow: bool = True,
                 clamp_max_for_overflow: bool = True,
                 visiual_dim: int = 256,
                 embed_dim: int = 2048,
                 n_head: int = 8,
                 vlfuse_use_checkpoint: bool = True) -> None:

        super(VLFuse, self).__init__()
        # common params
        self.lang_model = lang_model
        self.use_checkpoint = vlfuse_use_checkpoint

        # visiual params
        self.visiual_dim = visiual_dim

        # language params
        self.lang_dim = lang_dim
        # mha params
        self.n_head = n_head
        self.embed_dim = embed_dim  # 2048 by default

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.visiual_dim,  # 256
            l_dim=self.lang_dim,  # 768
            embed_dim=self.embed_dim,  # 2048
            num_heads=self.n_head,  # 8
            dropout=0.1,
            drop_path=.0,
            init_values=1.0 / 6,
            stable_softmax_2d=stable_softmax_2d,
            clamp_min_for_underflow=clamp_min_for_underflow,
            clamp_max_for_overflow=clamp_max_for_overflow)

    def forward(self, x, task=None):
        visual_features = x['visual']
        language_dict_features = x['lang']

        if self.use_checkpoint:
            fused_visual_features, language_features = checkpoint.checkpoint(
                self.b_attn, visual_features, language_dict_features['hidden'],
                language_dict_features['masks'], task)
        else:
            fused_visual_features, language_features = self.b_attn(
                visual_features, language_dict_features['hidden'],
                language_dict_features['masks'], task)

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {
            'visual': fused_visual_features,
            'lang': fused_language_dict_features
        }

        return features_dict


class VLTransformerEncoder(BaseModule):
    """Transformer encoder of Deformable DETR."""

    def __init__(self,
                 vlfuse_num_layers: int,
                 vlfuse_layer_cfg: ConfigType,
                 vision_num_layers: int,
                 vision_layer_cfg: ConfigType,
                 text_num_layers: int = 0,
                 text_layer_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.vlfuse_num_layers = vlfuse_num_layers
        self.vlfuse_layer_cfg = vlfuse_layer_cfg
        self.vision_num_layers = vision_num_layers
        self.vision_layer_cfg = vision_layer_cfg
        self.text_num_layers = text_num_layers
        self.text_layer_cfg = text_layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.vision_layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.vision_layer_cfg)
            for _ in range(self.vision_num_layers)
        ])
        if self.vlfuse_num_layers < self.vision_num_layers:
            self.add_vl_layers = self.vision_num_layers - \
                self.vlfuse_num_layers
        self.vlfuse_layers = ModuleList([
            VLFuse(**self.vlfuse_layer_cfg)
            for _ in range(self.vlfuse_num_layers)
        ])
        for _ in range(self.add_vl_layers):
            self.vlfuse_layers.append(nn.Identity())
        assert self.text_num_layers == 0
        if self.text_num_layers < self.vision_num_layers:
            self.add_text_layers = self.vision_num_layers \
                - self.text_num_layers
        self.text_layers = ModuleList(
            [nn.Identity() for _ in range(self.add_text_layers)])

        self.embed_dims = self.vision_layers[0].embed_dims

    def forward(self, language_dict_features: Tensor, query: Tensor,
                query_pos: Tensor, key_padding_mask: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        output = {
            'visual': query,
            'lang': language_dict_features,
        }
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for vl_layer, layer, lang_layer in zip(self.vlfuse_layers,
                                               self.vision_layers,
                                               self.text_layers):
            output = vl_layer(output)
            output['visual'] = layer(
                query=output['visual'],
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
            output = lang_layer(output)

        return output

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
