# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn

from mmdet.utils import OptMultiConfig
from mmdet.models.layers.transformer.utils import ConditionalAttention

class GroupAttention(ConditionalAttention):
    """A wrapper of conditional attention, dropout and residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop: A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        cross_attn (bool): Whether the attention module is for cross attention.
            Default: False
        keep_query_pos (bool): Whether to transform query_pos before cross
            attention.
            Default: False.
        batch_first (bool): When it is True, Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default: True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, *arg, group_detr=1, **kwargs) -> None:
        self.group_detr = group_detr
        super().__init__(*arg, **kwargs)


    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                attn_mask: Tensor = None,
                key_padding_mask: Tensor = None) -> Tensor:
        """Forward function for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None, it will
                be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None, it
                will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, embed_dims].
        """

        q_content = self.qcontent_proj(query)
        q_pos = self.qpos_proj(query_pos)
        k_content = self.kcontent_proj(query)
        k_pos = self.kpos_proj(query_pos)
        v = self.v_proj(query)
        q = q_content if q_pos is None else q_content + q_pos
        k = k_content if k_pos is None else k_content + k_pos
        num_queries, bs, _ = q_content.shape
        if self.training:
            q = torch.cat(
                q.split(num_queries // self.group_detr, dim=0), dim=1)
            k = torch.cat(
                k.split(num_queries // self.group_detr, dim=0), dim=1)
            v = torch.cat(
                v.split(num_queries // self.group_detr, dim=0), dim=1)
        sa_output = self.forward_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        if self.training:
            sa_output = torch.cat(sa_output.split(bs, dim=1), dim=0)
        query = query + self.proj_drop(sa_output)

        return query
