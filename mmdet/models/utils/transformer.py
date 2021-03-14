import copy

import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init)

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

    def forward(
        self,
        query,
        key=None,
        value=None,
        residual=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
    ):
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


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaluts to 2.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add resudual connection.
            Defaults to True.
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
    """Base class for vision transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        embed_dims (int): Embedding dimension of Transformerlayer.
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
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

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
            attn_mask (Tensor, optional): 2D Tensor used in
                calculation of corresponding attention.
                Defaults: None.
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
                    attn_mask=attn_mask,
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
                    attn_mask=attn_mask,
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
    """Implements one decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        embed_dims (int): Embedding dimension of Transformerlayer.
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
                attn_mask=None,
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
            attn_mask (Tensor, optional):  2D Tensor used in
                calculation of corresponding attention.
                Defaults: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_query].
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        for layer in self.layers:
            x = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=attn_mask,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return x


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


class DetrTransformerDecoder(BaseTransformerCoder):
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

        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.coder_norm = build_norm_layer(coder_norm_cfg, self.embed_dims)[1]

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.coder_norm:
                x = self.coder_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.coder_norm is not None:
                    intermediate.append(self.coder_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


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
            key=x,
            value=x,
            query_pos=pos_embed,
            attn_mask=None,
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
