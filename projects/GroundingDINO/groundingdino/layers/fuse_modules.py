import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from torch import Tensor


class BiMultiHeadAttention(nn.Module):
    """Bidirectional fusion Multi-Head Attention layer for Grounding DINO. It
    is a little bit different from that of GLIP. It also uses language
    attention mask in `forward` function.

    Args:
        v_dim (int): The dimension of the vision input.
        l_dim (int): The dimension of the language input.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f'embed_dim must be divisible by num_heads (got `embed_dim`: \
            {self.embed_dim} and `num_heads`: {self.num_heads}).'

        self.scale = self.head_dim**(-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
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

    def forward(self,
                vision: Tensor,
                lang: Tensor,
                attention_mask_v: Tensor = None,
                attention_mask_l: Tensor = None):
        """Forward pass of the BiMultiHeadAttention module.

        Args:
            visual (Tensor): Visual features. [bs, n_img, dim]
            lang (Tensor): Language features. [bs, n_text, dim]
            attention_mask_v (Tensor, optional): Visual attention
                mask. [bs, n_img]
            attention_mask_l (Tensor, optional): Text attention
                mask. [bs, n_text]
        """
        bsz, tgt_len, _ = vision.size()

        query_states = self.v_proj(vision) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(vision), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        # bs*nhead, nimg, ntxt
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

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
        attn_weights_l = attn_weights_T - \
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None,
                                 None, :].repeat(1, self.num_heads, 1,
                                                 1).flatten(0, 1))
            attn_weights_l.masked_fill_(attention_mask_v, float('-inf'))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None,
                                 None, :].repeat(1, self.num_heads, 1,
                                                 1).flatten(0, 1))
            attn_weights.masked_fill_(attention_mask_l, float('-inf'))
        attn_weights_v = attn_weights.softmax(dim=-1)

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


class BiAttentionBlock(nn.Module):

    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 drop_path: float = 0.0,
                 init_values: bool = 1e-4):
        """Bi-Direction MHA (text->image, image->text)

        Args:
            v_dim (int): Visual feature dimension.
            l_dim (int): Language feature dimension.
            embed_dim (int): Dimensionality of input and attention
                feature vectors.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout ratio.
            drop_path (float, optional): Droppath ratio.
            init_values (bool, optional): Gamma init values.
        """
        super(BiAttentionBlock, self).__init__()

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
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(
            init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(
            init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self,
                vision: Tensor,
                lang: Tensor,
                attention_mask_v: Tensor = None,
                attention_mask_l: Tensor = None):
        """Forward pass of the BiAttentionBlock module.

        Args:
            vision (Tensor): Visual features.
            lang (Tensor): Language features.
            attention_mask_v (Tensor, optional): Visual feature attention mask.
                Defaults to None.
            attention_mask_l (Tensor, optional): Text feature attention mask.
                Defaults to None.
        """
        vision = self.layer_norm_v(vision)
        lang = self.layer_norm_l(lang)
        delta_v, delta_l = self.attn(
            vision,
            lang,
            attention_mask_v=attention_mask_v,
            attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        vision = vision + self.drop_path(self.gamma_v * delta_v)
        lang = lang + self.drop_path(self.gamma_l * delta_l)
        return vision, lang
