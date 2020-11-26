import torch
import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init)

from .builder import TRANSFORMER


class MultiheadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default 0.0.
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
                x,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        """Forward function for `MultiheadAttention`.

        Args:
            x (Tensor): The input query with shape [num_query, bs,
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
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
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

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'dropout={self.dropout})'
        return repr_str


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        act_cfg (dict): The activation config for FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
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

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in DETR transformer.

    Args:
        embed_dims (int): The feature dimension. Same as `FFN`.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
        order (tuple[str]): The order for encoder layer. Valid examples are
            ('selfattn', 'norm', 'ffn', 'norm') and ('norm', 'selfattn',
            'norm', 'ffn'). Default ('selfattn', 'norm', 'ffn', 'norm').
        act_cfg (dict): The activation config for FFNs. Defalut ReLU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default 2.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2):
        super(TransformerEncoderLayer, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.pre_norm = order[0] == 'norm'
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout)
        self.ffn = FFN(embed_dims, feedforward_channels, num_fcs, act_cfg,
                       dropout)
        self.norms = nn.ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None):
        """Forward function for `TransformerEncoderLayer`.

        Args:
            x (Tensor): The input query with shape [num_key, bs,
                embed_dims]. Same in `MultiheadAttention.forward`.
            pos (Tensor): The positional encoding for query. Default None.
                Same as `query_pos` in `MultiheadAttention.forward`.
            attn_mask (Tensor): ByteTensor mask with shape [num_key,
                num_key]. Same in `MultiheadAttention.forward`. Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `MultiheadAttention.forward`. Default None.

        Returns:
            Tensor: forwarded results with shape [num_key, bs, embed_dims].
        """
        norm_cnt = 0
        inp_residual = x
        for layer in self.order:
            if layer == 'selfattn':
                # self attention
                query = key = value = x
                x = self.self_attn(
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos=pos,
                    key_pos=pos,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask)
                inp_residual = x
            elif layer == 'norm':
                x = self.norms[norm_cnt](x)
                norm_cnt += 1
            elif layer == 'ffn':
                x = self.ffn(x, inp_residual if self.pre_norm else None)
        return x

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'order={self.order}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs})'
        return repr_str


class TransformerDecoderLayer(nn.Module):
    """Implements one decoder layer in DETR transformer.

    Args:
        embed_dims (int): The feature dimension. Same as
            `TransformerEncoderLayer`.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): Same as `TransformerEncoderLayer`.
        dropout (float): Same as `TransformerEncoderLayer`. Default 0.0.
        order (tuple[str]): The order for decoder layer. Valid examples are
            ('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn', 'norm') and
            ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn').
            Default the former.
        act_cfg (dict): Same as `TransformerEncoderLayer`. Defalut ReLU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        num_fcs (int): The number of fully-connected layers in FFNs.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                        'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2):
        super(TransformerDecoderLayer, self).__init__()
        assert isinstance(order, tuple) and len(order) == 6
        assert set(order) == set(['selfattn', 'norm', 'multiheadattn', 'ffn'])
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.pre_norm = order[0] == 'norm'
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout)
        self.multihead_attn = MultiheadAttention(embed_dims, num_heads,
                                                 dropout)
        self.ffn = FFN(embed_dims, feedforward_channels, num_fcs, act_cfg,
                       dropout)
        self.norms = nn.ModuleList()
        # 3 norm layers in official DETR's TransformerDecoderLayer
        for _ in range(3):
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

    def forward(self,
                x,
                memory,
                memory_pos=None,
                query_pos=None,
                memory_attn_mask=None,
                target_attn_mask=None,
                memory_key_padding_mask=None,
                target_key_padding_mask=None):
        """Forward function for `TransformerDecoderLayer`.

        Args:
            x (Tensor): Input query with shape [num_query, bs, embed_dims].
            memory (Tensor): Tensor got from `TransformerEncoder`, with shape
                [num_key, bs, embed_dims].
            memory_pos (Tensor): The positional encoding for `memory`. Default
                None. Same as `key_pos` in `MultiheadAttention.forward`.
            query_pos (Tensor): The positional encoding for `query`. Default
                None. Same as `query_pos` in `MultiheadAttention.forward`.
            memory_attn_mask (Tensor): ByteTensor mask for `memory`, with
                shape [num_key, num_key]. Same as `attn_mask` in
                `MultiheadAttention.forward`. Default None.
            target_attn_mask (Tensor): ByteTensor mask for `x`, with shape
                [num_query, num_query]. Same as `attn_mask` in
                `MultiheadAttention.forward`. Default None.
            memory_key_padding_mask (Tensor): ByteTensor for `memory`, with
                shape [bs, num_key]. Same as `key_padding_mask` in
                `MultiheadAttention.forward`. Default None.
            target_key_padding_mask (Tensor): ByteTensor for `x`, with shape
                [bs, num_query]. Same as `key_padding_mask` in
                `MultiheadAttention.forward`. Default None.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        norm_cnt = 0
        inp_residual = x
        for layer in self.order:
            if layer == 'selfattn':
                query = key = value = x
                x = self.self_attn(
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos,
                    key_pos=query_pos,
                    attn_mask=target_attn_mask,
                    key_padding_mask=target_key_padding_mask)
                inp_residual = x
            elif layer == 'norm':
                x = self.norms[norm_cnt](x)
                norm_cnt += 1
            elif layer == 'multiheadattn':
                query = x
                key = value = memory
                x = self.multihead_attn(
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos,
                    key_pos=memory_pos,
                    attn_mask=memory_attn_mask,
                    key_padding_mask=memory_key_padding_mask)
                inp_residual = x
            elif layer == 'ffn':
                x = self.ffn(x, inp_residual if self.pre_norm else None)
        return x

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'order={self.order}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs})'
        return repr_str


class TransformerEncoder(nn.Module):
    """Implements the encoder in DETR transformer.

    Args:
        num_layers (int): The number of `TransformerEncoderLayer`.
        embed_dims (int): Same as `TransformerEncoderLayer`.
        num_heads (int): Same as `TransformerEncoderLayer`.
        feedforward_channels (int): Same as `TransformerEncoderLayer`.
        dropout (float): Same as `TransformerEncoderLayer`. Default 0.0.
        order (tuple[str]): Same as `TransformerEncoderLayer`.
        act_cfg (dict): Same as `TransformerEncoderLayer`. Defalut ReLU.
        norm_cfg (dict): Same as `TransformerEncoderLayer`. Default
            layer normalization.
        num_fcs (int): Same as `TransformerEncoderLayer`. Default 2.
    """

    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2):
        super(TransformerEncoder, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.pre_norm = order[0] == 'norm'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims, num_heads,
                                        feedforward_channels, dropout, order,
                                        act_cfg, norm_cfg, num_fcs))
        self.norm = build_norm_layer(norm_cfg,
                                     embed_dims)[1] if self.pre_norm else None

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None):
        """Forward function for `TransformerEncoder`.

        Args:
            x (Tensor): Input query. Same in `TransformerEncoderLayer.forward`.
            pos (Tensor): Positional encoding for query. Default None.
                Same in `TransformerEncoderLayer.forward`.
            attn_mask (Tensor): ByteTensor attention mask. Default None.
                Same in `TransformerEncoderLayer.forward`.
            key_padding_mask (Tensor): Same in
                `TransformerEncoderLayer.forward`. Default None.

        Returns:
            Tensor: Results with shape [num_key, bs, embed_dims].
        """
        for layer in self.layers:
            x = layer(x, pos, attn_mask, key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_layers={self.num_layers}, '
        repr_str += f'embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'order={self.order}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs})'
        return repr_str


class TransformerDecoder(nn.Module):
    """Implements the decoder in DETR transformer.

    Args:
        num_layers (int): The number of `TransformerDecoderLayer`.
        embed_dims (int): Same as `TransformerDecoderLayer`.
        num_heads (int): Same as `TransformerDecoderLayer`.
        feedforward_channels (int): Same as `TransformerDecoderLayer`.
        dropout (float): Same as `TransformerDecoderLayer`. Default 0.0.
        order (tuple[str]): Same as `TransformerDecoderLayer`.
        act_cfg (dict): Same as `TransformerDecoderLayer`. Defalut ReLU.
        norm_cfg (dict): Same as `TransformerDecoderLayer`. Default
            layer normalization.
        num_fcs (int): Same as `TransformerDecoderLayer`. Default 2.
    """

    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                        'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        assert isinstance(order, tuple) and len(order) == 6
        assert set(order) == set(['selfattn', 'norm', 'multiheadattn', 'ffn'])
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(embed_dims, num_heads,
                                        feedforward_channels, dropout, order,
                                        act_cfg, norm_cfg, num_fcs))
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self,
                x,
                memory,
                memory_pos=None,
                query_pos=None,
                memory_attn_mask=None,
                target_attn_mask=None,
                memory_key_padding_mask=None,
                target_key_padding_mask=None):
        """Forward function for `TransformerDecoder`.

        Args:
            x (Tensor): Input query. Same in `TransformerDecoderLayer.forward`.
            memory (Tensor): Same in `TransformerDecoderLayer.forward`.
            memory_pos (Tensor): Same in `TransformerDecoderLayer.forward`.
                Default None.
            query_pos (Tensor): Same in `TransformerDecoderLayer.forward`.
                Default None.
            memory_attn_mask (Tensor): Same in
                `TransformerDecoderLayer.forward`. Default None.
            target_attn_mask (Tensor): Same in
                `TransformerDecoderLayer.forward`. Default None.
            memory_key_padding_mask (Tensor): Same in
                `TransformerDecoderLayer.forward`. Default None.
            target_key_padding_mask (Tensor): Same in
                `TransformerDecoderLayer.forward`. Default None.

        Returns:
            Tensor: Results with shape [num_query, bs, embed_dims].
        """
        intermediate = []
        for layer in self.layers:
            x = layer(x, memory, memory_pos, query_pos, memory_attn_mask,
                      target_attn_mask, memory_key_padding_mask,
                      target_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(x))
        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x.unsqueeze(0)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_layers={self.num_layers}, '
        repr_str += f'embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'order={self.order}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'return_intermediate={self.return_intermediate})'
        return repr_str


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
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        num_encoder_layers (int): Number of `TransformerEncoderLayer`.
        num_decoder_layers (int): Number of `TransformerDecoderLayer`.
        feedforward_channels (int): The hidden dimension for FFNs used in both
            encoder and decoder.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
        act_cfg (dict): Activation config for FFNs used in both encoder
            and decoder. Defalut ReLU.
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
                 embed_dims=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 feedforward_channels=2048,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2,
                 pre_norm=False,
                 return_intermediate_dec=False):
        super(Transformer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.num_fcs = num_fcs
        self.pre_norm = pre_norm
        self.return_intermediate_dec = return_intermediate_dec
        if self.pre_norm:
            encoder_order = ('norm', 'selfattn', 'norm', 'ffn')
            decoder_order = ('norm', 'selfattn', 'norm', 'multiheadattn',
                             'norm', 'ffn')
        else:
            encoder_order = ('selfattn', 'norm', 'ffn', 'norm')
            decoder_order = ('selfattn', 'norm', 'multiheadattn', 'norm',
                             'ffn', 'norm')
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dims,
                                          num_heads, feedforward_channels,
                                          dropout, encoder_order, act_cfg,
                                          norm_cfg, num_fcs)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dims,
                                          num_heads, feedforward_channels,
                                          dropout, decoder_order, act_cfg,
                                          norm_cfg, num_fcs,
                                          return_intermediate_dec)

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
            x, pos=pos_embed, attn_mask=None, key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            target,
            memory,
            memory_pos=pos_embed,
            query_pos=query_embed,
            memory_attn_mask=None,
            target_attn_mask=None,
            memory_key_padding_mask=mask,
            target_key_padding_mask=None)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'num_encoder_layers={self.num_encoder_layers}, '
        repr_str += f'num_decoder_layers={self.num_decoder_layers}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'norm_cfg={self.norm_cfg}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'pre_norm={self.pre_norm}, '
        repr_str += f'return_intermediate_dec={self.return_intermediate_dec})'
        return repr_str
