import copy
import math
import warnings

import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      constant_init, xavier_init)
from mmcv.ops.deformable_attention import MultiScaleDeformAttnFunction

from .builder import (ATTENTION, TRANSFORMER, TRANSFORMERLAYER,
                      build_attention, build_transformerlayer)


@ATTENTION.register_module()
class MultiheadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        Args:
            query (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. Default None. If not None, it will
                be added to `x` before forward function.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Default None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (Tensor): ByteTensor mask with shape [num_query,
                num_key]. Same in `nn.MultiheadAttention.forward`.
                Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `nn.MultiheadAttention.forward`. Default None.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = query
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(
            query,
            key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return residual + self.dropout(out)


@ATTENTION.register_module()
class MultiScaleDeformableAttention(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    # TODO change to init_weight
    def init_weight(self):
        constant_init(self.sampling_offsets, 0.)
        # constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_init(self.attention_weights, val=0., bias=0.)
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

    # TODO check the shape of KQV
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        assert residual is None, 'MultiScaleDeformableAttention dose not' \
                                 'support add residual'

        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # TODO: check value why so many zeros
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.num_heads,
                           self.embed_dims // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points)
        attention_weights = attention_weights.softmax(-1)
        # N, Len_q, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points * \
                reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        output = MultiScaleDeformAttnFunction.apply(value, spatial_shapes,
                                                    level_start_index,
                                                    sampling_locations,
                                                    attention_weights,
                                                    self.im2col_step)
        output = self.output_proj(output).permute(1, 0, 2)
        # (num_query, bs ,embed_dims)
        return output


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: `ReLU`
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add residual connection.
            Default: `True`.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)


@TRANSFORMERLAYER.register_module()
class BaseTransformerLayer(nn.Module):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more customization,
    for example, using any number of `FFN or LN ` and use different kinds
    of `attention` by specifying a list of `ConfigDict` named `attn_cfgs`.
    It is worth mentioning that it supports `prenorm` when you specifying
    `norm` as the first element of `operation_order`. More details about
    the `prenorm`: `On Layer Normalization in the Transformer Architecture
    <https://arxiv.org/abs/2007.08103>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`. Default: None.
        feedforward_channels (int): The hidden dimension for FFNs.
            Default: None.
        embed_dims (int): Embedding dimension of Transformerlayer.
            Default: 256.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('selfattn', 'norm', 'ffn', 'norm').
            Support prenorm when you specifying first element as `norm`.
            Default：None.
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
            self,
            attn_cfgs=None,
            feedforward_channels=None,
            embed_dims=256,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
    ):

        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order) & set(
            ['selfattn', 'norm', 'ffn', 'crossattn']) == set(operation_order)
        num_attn = operation_order.count('selfattn') + operation_order.count(
            'crossattn')
        attn_cfgs = copy.deepcopy(attn_cfgs)
        if isinstance(attn_cfgs, ConfigDict):
            attn_cfgs = [attn_cfgs for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'
        assert 'embed_dims' in attn_cfgs[0]

        self.num_attn = num_attn
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.ffn_num_fcs = ffn_num_fcs
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        index = 0
        for operation in operation_order:
            if operation in ['selfattn', 'crossattn']:
                attention = build_attention(attn_cfgs[index])
                self.attentions.append(attention)
                index += 1

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(
                FFN(self.embed_dims, feedforward_channels, ffn_num_fcs,
                    act_cfg, ffn_dropout))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        Args:
            query (Tensor): Input query with the shape
                `(num_query, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Defaults: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_query]. Only used in `selfattn` layer.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        inp_residual = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'selfattn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'crossattn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, inp_residual if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMERLAYER.register_module()
class DetrTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        embed_dims (int): Embedding dimension of Transformerlayer.
            Default: 256.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('selfattn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs.
        norm_cfg (dict): Config dict for normalization layer.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
            self,
            attn_cfgs,
            feedforward_channels,
            embed_dims=256,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
    ):
        super(DetrTransformerEncoderLayer, self).__init__(
            embed_dims=embed_dims,
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            operation_order=operation_order,
            ffn_dropout=ffn_dropout,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
        )
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['selfattn', 'norm', 'ffn'])


@TRANSFORMERLAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        embed_dims (int): Embedding dimension of Transformerlayer.
            Default: 256
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('selfattn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
            self,
            attn_cfgs,
            feedforward_channels,
            embed_dims=256,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            embed_dims=embed_dims,
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
        )
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['selfattn', 'norm', 'crossattn', 'ffn'])


class BaseTransformerCoder(nn.Module):
    """Base coder in vision transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `tranformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None):
        super(BaseTransformerCoder, self).__init__()
        if isinstance(transformerlayers, ConfigDict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        self.num_layers = num_layers
        assert 'embed_dims' in transformerlayers[0]
        self.embed_dims = transformerlayers[0].get('embed_dims')
        operation_order = transformerlayers[0]['operation_order']
        self.pre_norm = operation_order[0] == 'norm'
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformerlayer(transformerlayers[i]))

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Defaults: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_query]. Only used in self-attention
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query


class DetrTransformerEncoder(BaseTransformerCoder):
    """TransformerEncoder of DETR.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
            self,
            *args,
            coder_norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        self.coder_norm = build_norm_layer(
            coder_norm_cfg, self.embed_dims)[1] if self.pre_norm else None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.coder_norm is not None:
            x = self.coder_norm(x)
        return x


class DetrTransformerDecoder():
    pass


class DeformableDetrTransformerDecoder(BaseTransformerCoder):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 coder_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(DeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.coder_norm = build_norm_layer(coder_norm_cfg, self.embed_dims)[1]
        # TODO make this when as two stage
        self.bbox_embed = None

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            # hack implementation for iterative bounding box refinement
            def inverse_sigmoid(x, eps=1e-5):
                x = x.clamp(min=0, max=1)
                x1 = x.clamp(min=eps)
                x2 = (1 - x).clamp(min=eps)
                return torch.log(x1 / x2)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            # TODO remove this
            output = output.permute(1, 0, 2)

            if self.return_intermediate:

                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class Transformer(nn.Module):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        attn_cfgs (sequence[obj:`mmcv.Config`]) : Config of
            Attention in Encoder and decoder in Transformer.
        num_encoder_layers (int): Number of `TransformerEncoderLayer`.
        num_decoder_layers (int): Number of `TransformerDecoderLayer`.
        feedforward_channels (int): The hidden dimension for FFNs used in both
            encoder and decoder.
        ffn_dropout (float): Probability of an element to
            be zeroed. Default 0.0.
        act_cfg (dict): Activation config for FFNs used in both encoder
            and decoder. Defalut: ReLU.
        norm_cfg (dict): Config dict for normalization used in both encoder
            and decoder. Default layer normalization.
        num_fcs (int): The number of fully-connected layers in FFNs, which is
            used for both encoder and decoder.
        pre_norm (bool): Whether the normalization layer is ordered
            first in the encoder and decoder. Default False.
        return_intermediate_dec (bool): Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False. If False, the returned
            `hs` has shape [num_decoder_layers, bs, num_query, embed_dims].
            If True, the returned `hs` will have shape [1, bs, num_query,
            embed_dims].
    """

    def __init__(self,
                 encodelayers=None,
                 dencodelayers=None,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 norm_cfg=dict(type='LN'),
                 return_intermediate=False):
        super(Transformer, self).__init__()
        encodelayers = copy.deepcopy(encodelayers)
        dncodelayers = copy.deepcopy(dencodelayers)
        if isinstance(encodelayers, ConfigDict):
            encodelayers = [
                copy.deepcopy(encodelayers) for _ in range(num_encoder_layers)
            ]
        if isinstance(dencodelayers, ConfigDict):
            dencodelayers = [
                copy.deepcopy(dncodelayers) for _ in range(num_encoder_layers)
            ]

        self.encoder = DetrTransformerEncoder(
            transformerlayers=encodelayers,
            num_layers=num_encoder_layers,
            coder_norm_cfg=norm_cfg,
        )
        self.decoder = DetrTransformerDecoder(
            transformerlayers=dencodelayers,
            num_layers=num_decoder_layers,
            coder_norm_cfg=norm_cfg,
            return_intermediate=return_intermediate)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution=distribution)

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.flatten(1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory


@TRANSFORMER.register_module()
class DeformableDetrTransformer(nn.Module):
    """Implements the Deformable Detr  transformer.

    Args:
        attn_cfgs (sequence[obj:`mmcv.Config`]) : Config of
            Attention in Encoder and decoder in Transformer.
        num_encoder_layers (int): Number of `TransformerEncoderLayer`.
        num_decoder_layers (int): Number of `TransformerDecoderLayer`.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4
        feedforward_channels (int): The hidden dimension for FFNs used in both
            encoder and decoder.
        ffn_dropout (float): Probability of an element to
            be zeroed. Default 0.0.
        act_cfg (dict): Activation config for FFNs used in both encoder
            and decoder. Defalut: ReLU.
        norm_cfg (dict): Config dict for normalization used in both encoder
            and decoder. Default layer normalization.
        num_fcs (int): The number of fully-connected layers in FFNs, which is
            used for both encoder and decoder.
        pre_norm (bool): Whether the normalization layer is ordered
            first in the encoder and decoder. Default False.
        return_intermediate_dec (bool): Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False. If False, the returned
            `hs` has shape [num_decoder_layers, bs, num_query, embed_dims].
            If True, the returned `hs` will have shape [1, bs, num_query,
            embed_dims].
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 encodelayers=None,
                 dencodelayers=None,
                 embed_dims=256,
                 as_two_stage=False,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_feature_levels=4,
                 norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 two_stage_num_proposals=300):
        super(DeformableDetrTransformer, self).__init__()
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        self.embed_dims = embed_dims
        self.two_stage_num_proposals = two_stage_num_proposals

        encodelayers = copy.deepcopy(encodelayers)
        dncodelayers = copy.deepcopy(dencodelayers)
        if isinstance(encodelayers, ConfigDict):
            encodelayers = [
                copy.deepcopy(encodelayers) for _ in range(num_encoder_layers)
            ]
        if isinstance(dencodelayers, ConfigDict):
            dencodelayers = [
                copy.deepcopy(dncodelayers) for _ in range(num_encoder_layers)
            ]
        self.encoder = DetrTransformerEncoder(
            transformerlayers=encodelayers,
            num_layers=num_encoder_layers,
            coder_norm_cfg=norm_cfg,
        )
        self.decoder = DeformableDetrTransformerDecoder(
            transformerlayers=dencodelayers,
            num_layers=num_decoder_layers,
            coder_norm_cfg=norm_cfg,
            return_intermediate=return_intermediate)

        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution=distribution)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        assert self.as_two_stage or query_embed is not None
        # TODO: make all to (num_query, bs, num_dims)
        # prepare input for encoder
        # TODO : move these to encoder may be better ?
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        # encoder
        # TODO change the dim order
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        # prepare input for decoder
        # move these to decoder may be batter ?
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact = \
                self.decoder.bbox_embed[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embeds, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embeds, tgt = torch.split(query_embed, c, dim=1)
            query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embeds).sigmoid()
            init_reference_out = reference_points

        # decoder
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_embeds = query_embeds.permute(1, 0, 2)
        hs, inter_references = self.decoder(
            query=tgt,
            key=None,
            value=memory,
            pos_query=query_embeds,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        inter_references_out = inter_references
        if self.as_two_stage:
            return hs, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        return hs, init_reference_out, \
            inter_references_out, None, None


@TRANSFORMER.register_module()
class DynamicConv(nn.Module):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        self.fc_layer = nn.Linear(num_output, self.out_channels)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        num_proposals = param_feature.size(0)
        input_feature = input_feature.view(num_proposals, self.in_channels,
                                           -1).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features
