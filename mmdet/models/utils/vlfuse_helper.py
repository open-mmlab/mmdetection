# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint

try:
    from transformers import BertPreTrainedModel
    from transformers.activations import ACT2FN
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:
    BertPreTrainedModel = None
    ACT2FN = None
    apply_chunking_to_forward = None

def permute_and_flatten(layer, N, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def clamp_values(vector, min_val=-50000, max_val=50000):
    vector = torch.clamp(vector, min=min_val, max=max_val)
    return vector


class BiMultiHeadAttention(nn.Module):

    def __init__(self,
                 v_dim,
                 l_dim,
                 embed_dim,
                 num_heads,
                 dropout=0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim**(-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

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

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}'
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

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

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}'
                )
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
            raise ValueError(
                f'`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}'
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len,
                                    self.head_dim):
            raise ValueError(
                f'`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}'
            )

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
                 cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
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
            dropout=dropout)

        # add layer scale for training stability
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(
            init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(
            init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg

    def forward(self,
                q0,
                q1,
                q2,
                q3,
                q4,
                l,
                attention_mask_l=None):

        visu_feat = []
        size_per_level, visual_features_flatten = [], []
        for i, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = torch.cat(visual_features_flatten, dim=1)
        new_v, new_l = self.single_attention_call(
            visual_features_flatten, l, attention_mask_l=attention_mask_l)
        # [bs, N, C] -> [bs, C, N]
        new_v = new_v.transpose(1, 2).contiguous()

        start = 0
        for (h, w) in size_per_level:
            new_v_per_level = new_v[:, :,
                                    start:start + h * w].view(bs, -1, h,
                                                              w).contiguous()
            visu_feat.append(new_v_per_level)
            start += h * w

        lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], \
            lang_feat[2], lang_feat[3], lang_feat[4]

    def single_attention_call(self,
                              v,
                              l,
                              attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


class VLFuse(torch.nn.Module):
    """Early Fusion Module."""

    def __init__(self):
        super(VLFuse, self).__init__()
        self.init_configs()

        print('EARLY FUSION ON, USING {}'.format('MHA-B'))

        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.joint_embedding_size,
            l_dim=self.lang_dim,
            embed_dim=self.embed_dim,
            num_heads=self.n_head,
            dropout=0.1,
            drop_path=.0,
            init_values=1.0 / 6.0)

    def init_configs(self):
        # common params
        self.lang_model = 'bert-base-uncased'
        self.joint_embedding_size = 256
        self.joint_embedding_dropout = 0.1
        self.joint_mlp_layers = 2

        self.max_query_len = 256
        self.n_layers = 1
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = 256

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        self.lang_dim = 768

    def forward(self, x):
        visual_features = x['visual']
        language_dict_features = x['lang']

        fused_visual_features = None
        fused_language_dict_features = None

        q0, q1, q2, q3, q4, l0, _, _, _, _ = checkpoint.checkpoint(
                self.b_attn, visual_features[0], visual_features[1],
                visual_features[2], visual_features[3], visual_features[4],
                language_dict_features['hidden'],
                language_dict_features['masks'])

        fused_visual_features = [q0, q1, q2, q3, q4]
        language_features = l0

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {
            'visual': fused_visual_features,
            'lang': fused_language_dict_features
        }

        return features_dict


# Bert related classes Used for VLFusion
class BertSelfAttention(nn.Module):

    def __init__(self,
                 config,
                 clamp_min_for_underflow=False,
                 clamp_max_for_overflow=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    'bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        if self.clamp_min_for_underflow:
            attention_scores = torch.clamp(
                attention_scores, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attention_scores = torch.clamp(
                attention_scores, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # if math.isnan(attention_probs.sum().item()):
        #     for i in range(attention_probs.size(1)):
        #         for j in range(attention_probs.size(2)):
        #             if math.isnan(attention_probs[0, i, j].sum().item()):
        #                 print(i, j)
        #                 pdb.set_trace()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self,
                 config,
                 clamp_min_for_underflow=False,
                 clamp_max_for_overflow=False):
        super().__init__()
        self.self = BertSelfAttention(config, clamp_min_for_underflow,
                                      clamp_max_for_overflow)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads,
            self.self.attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = clamp_values(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = clamp_values(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = clamp_values(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = clamp_values(hidden_states)
        return hidden_states


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
        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it
        # broadcastable to all heads.
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
        outputs = (layer_output,) + outputs
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
