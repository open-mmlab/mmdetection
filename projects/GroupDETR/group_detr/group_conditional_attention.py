# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdet.models.layers.transformer.utils import ConditionalAttention


class GroupConditionalAttention(ConditionalAttention):
    """A wrapper of conditional attention in GroupDETR.

    Args:
        num_query_groups (int): The number of decoder query groups.
    """

    def __init__(self, *arg, num_query_groups: int = 1, **kwargs) -> None:
        self.num_query_groups = num_query_groups
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
        bs, num_queries, _ = q_content.shape
        # split the qkv groups of decoder self-attention
        if self.training:
            q = torch.cat(
                q.split(num_queries // self.num_query_groups, dim=1), dim=0)
            k = torch.cat(
                k.split(num_queries // self.num_query_groups, dim=1), dim=0)
            v = torch.cat(
                v.split(num_queries // self.num_query_groups, dim=1), dim=0)
        sa_output = self.forward_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        # concat the output of decoder self-attention
        if self.training:
            sa_output = torch.cat(sa_output.split(bs, dim=0), dim=1)
        query = query + self.proj_drop(sa_output)

        return query
