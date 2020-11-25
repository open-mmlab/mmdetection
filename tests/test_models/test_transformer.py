from unittest.mock import patch

import pytest
import torch

from mmdet.models.utils import (FFN, MultiheadAttention, Transformer,
                                TransformerDecoder, TransformerDecoderLayer,
                                TransformerEncoder, TransformerEncoderLayer)


def _ffn_forward(self, x, residual=None):
    if residual is None:
        residual = x
    residual_str = residual.split('_')[-1]
    if '(residual' in residual_str:
        residual_str = residual_str.split('(residual')[0]
    return x + '_ffn(residual={})'.format(residual_str)


def _multihead_attention_forward(self,
                                 x,
                                 key=None,
                                 value=None,
                                 residual=None,
                                 query_pos=None,
                                 key_pos=None,
                                 attn_mask=None,
                                 key_padding_mask=None,
                                 selfattn=True):
    if residual is None:
        residual = x
    residual_str = residual.split('_')[-1]
    if '(residual' in residual_str:
        residual_str = residual_str.split('(residual')[0]
    attn_str = 'selfattn' if selfattn else 'multiheadattn'
    return x + '_{}(residual={})'.format(attn_str, residual_str)


def _encoder_layer_forward(self,
                           x,
                           pos=None,
                           attn_mask=None,
                           key_padding_mask=None):
    norm_cnt = 0
    inp_residual = x
    for layer in self.order:
        if layer == 'selfattn':
            x = self.self_attn(
                x,
                x,
                x,
                inp_residual if self.pre_norm else None,
                query_pos=pos,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)
            inp_residual = x
        elif layer == 'norm':
            x = x + '_norm{}'.format(norm_cnt)
            norm_cnt += 1
        elif layer == 'ffn':
            x = self.ffn(x, inp_residual if self.pre_norm else None)
        else:
            raise ValueError(f'Unsupported layer type {layer}.')
    return x


def _decoder_layer_forward(self,
                           x,
                           memory,
                           memory_pos=None,
                           query_pos=None,
                           memory_attn_mask=None,
                           target_attn_mask=None,
                           memory_key_padding_mask=None,
                           target_key_padding_mask=None):
    norm_cnt = 0
    inp_residual = x
    for layer in self.order:
        if layer == 'selfattn':
            x = self.self_attn(
                x,
                x,
                x,
                inp_residual if self.pre_norm else None,
                query_pos,
                attn_mask=target_attn_mask,
                key_padding_mask=target_key_padding_mask)
            inp_residual = x
        elif layer == 'norm':
            x = x + '_norm{}'.format(norm_cnt)
            norm_cnt += 1
        elif layer == 'multiheadattn':
            x = self.multihead_attn(
                x,
                memory,
                memory,
                inp_residual if self.pre_norm else None,
                query_pos,
                key_pos=memory_pos,
                attn_mask=memory_attn_mask,
                key_padding_mask=memory_key_padding_mask,
                selfattn=False)
            inp_residual = x
        elif layer == 'ffn':
            x = self.ffn(x, inp_residual if self.pre_norm else None)
        else:
            raise ValueError(f'Unsupported layer type {layer}.')
    return x


def test_multihead_attention(embed_dims=8,
                             num_heads=2,
                             dropout=0.1,
                             num_query=5,
                             num_key=10,
                             batch_size=1):
    module = MultiheadAttention(embed_dims, num_heads, dropout)
    # self attention
    query = torch.rand(num_query, batch_size, embed_dims)
    out = module(query)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set key
    key = torch.rand(num_key, batch_size, embed_dims)
    out = module(query, key)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set residual
    residual = torch.rand(num_query, batch_size, embed_dims)
    out = module(query, key, key, residual)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set query_pos and key_pos
    query_pos = torch.rand(num_query, batch_size, embed_dims)
    key_pos = torch.rand(num_key, batch_size, embed_dims)
    out = module(query, key, None, residual, query_pos, key_pos)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set key_padding_mask
    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(query, key, None, residual, query_pos, key_pos, None,
                 key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set attn_mask
    attn_mask = torch.rand(num_query, num_key) > 0.5
    out = module(query, key, key, residual, query_pos, key_pos, attn_mask,
                 key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)


def test_ffn(embed_dims=8, feedforward_channels=8, num_fcs=2, batch_size=1):
    # test invalid num_fcs
    with pytest.raises(AssertionError):
        module = FFN(embed_dims, feedforward_channels, 1)

    module = FFN(embed_dims, feedforward_channels, num_fcs)
    x = torch.rand(batch_size, embed_dims)
    out = module(x)
    assert out.shape == (batch_size, embed_dims)
    # set residual
    residual = torch.rand(batch_size, embed_dims)
    out = module(x, residual)
    assert out.shape == (batch_size, embed_dims)

    # test case with no residual
    module = FFN(embed_dims, feedforward_channels, num_fcs, add_residual=False)
    x = torch.rand(batch_size, embed_dims)
    out = module(x)
    assert out.shape == (batch_size, embed_dims)


def test_transformer_encoder_layer(embed_dims=8,
                                   num_heads=2,
                                   feedforward_channels=8,
                                   num_key=10,
                                   batch_size=1):
    x = torch.rand(num_key, batch_size, embed_dims)
    # test invalid number of order
    with pytest.raises(AssertionError):
        order = ('norm', 'selfattn', 'norm', 'ffn', 'norm')
        module = TransformerEncoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)

    # test invalid value of order
    with pytest.raises(AssertionError):
        order = ('norm', 'selfattn', 'norm', 'unknown')
        module = TransformerEncoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)

    module = TransformerEncoderLayer(embed_dims, num_heads,
                                     feedforward_channels)

    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(x, key_padding_mask=key_padding_mask)
    assert not module.pre_norm
    assert out.shape == (num_key, batch_size, embed_dims)

    # set pos
    pos = torch.rand(num_key, batch_size, embed_dims)
    out = module(x, pos, key_padding_mask=key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)

    # set attn_mask
    attn_mask = torch.rand(num_key, num_key) > 0.5
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)

    # set pre_norm
    order = ('norm', 'selfattn', 'norm', 'ffn')
    module = TransformerEncoderLayer(
        embed_dims, num_heads, feedforward_channels, order=order)
    assert module.pre_norm
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)

    @patch('mmdet.models.utils.TransformerEncoderLayer.forward',
           _encoder_layer_forward)
    @patch('mmdet.models.utils.FFN.forward', _ffn_forward)
    @patch('mmdet.models.utils.MultiheadAttention.forward',
           _multihead_attention_forward)
    def test_order():
        module = TransformerEncoderLayer(embed_dims, num_heads,
                                         feedforward_channels)
        out = module('input')
        assert out == 'input_selfattn(residual=input)_norm0_ffn' \
            '(residual=norm0)_norm1'

        # pre_norm
        order = ('norm', 'selfattn', 'norm', 'ffn')
        module = TransformerEncoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)
        out = module('input')
        assert out == 'input_norm0_selfattn(residual=input)_' \
            'norm1_ffn(residual=selfattn)'

    test_order()


def test_transformer_decoder_layer(embed_dims=8,
                                   num_heads=2,
                                   feedforward_channels=8,
                                   num_key=10,
                                   num_query=5,
                                   batch_size=1):
    query = torch.rand(num_query, batch_size, embed_dims)
    # test invalid number of order
    with pytest.raises(AssertionError):
        order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                 'norm')
        module = TransformerDecoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)

    # test invalid value of order
    with pytest.raises(AssertionError):
        order = ('norm', 'selfattn', 'unknown', 'multiheadattn', 'norm', 'ffn')
        module = TransformerDecoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)

    module = TransformerDecoderLayer(embed_dims, num_heads,
                                     feedforward_channels)
    memory = torch.rand(num_key, batch_size, embed_dims)
    assert not module.pre_norm
    out = module(query, memory)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set query_pos
    query_pos = torch.rand(num_query, batch_size, embed_dims)
    out = module(query, memory, memory_pos=None, query_pos=query_pos)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set memory_pos
    memory_pos = torch.rand(num_key, batch_size, embed_dims)
    out = module(query, memory, memory_pos, query_pos)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set memory_key_padding_mask
    memory_key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set target_key_padding_mask
    target_key_padding_mask = torch.rand(batch_size, num_query) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask,
        target_key_padding_mask=target_key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set memory_attn_mask
    memory_attn_mask = torch.rand(num_query, num_key)
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_attn_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        target_key_padding_mask=target_key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    # set target_attn_mask
    target_attn_mask = torch.rand(num_query, num_query)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 target_attn_mask, memory_key_padding_mask,
                 target_key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    # pre_norm
    order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
    module = TransformerDecoderLayer(
        embed_dims, num_heads, feedforward_channels, order=order)
    assert module.pre_norm
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_attn_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        target_key_padding_mask=target_key_padding_mask)
    assert out.shape == (num_query, batch_size, embed_dims)

    @patch('mmdet.models.utils.TransformerDecoderLayer.forward',
           _decoder_layer_forward)
    @patch('mmdet.models.utils.FFN.forward', _ffn_forward)
    @patch('mmdet.models.utils.MultiheadAttention.forward',
           _multihead_attention_forward)
    def test_order():
        module = TransformerDecoderLayer(embed_dims, num_heads,
                                         feedforward_channels)
        out = module('input', 'memory')
        assert out == 'input_selfattn(residual=input)_norm0_multiheadattn' \
            '(residual=norm0)_norm1_ffn(residual=norm1)_norm2'

        # pre_norm
        order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
        module = TransformerDecoderLayer(
            embed_dims, num_heads, feedforward_channels, order=order)
        out = module('input', 'memory')
        assert out == 'input_norm0_selfattn(residual=input)_norm1_' \
            'multiheadattn(residual=selfattn)_norm2_ffn(residual=' \
            'multiheadattn)'

    test_order()


def test_transformer_encoder(num_layers=2,
                             embed_dims=8,
                             num_heads=2,
                             feedforward_channels=8,
                             num_key=10,
                             batch_size=1):
    module = TransformerEncoder(num_layers, embed_dims, num_heads,
                                feedforward_channels)
    assert not module.pre_norm
    assert module.norm is None
    x = torch.rand(num_key, batch_size, embed_dims)
    out = module(x)
    assert out.shape == (num_key, batch_size, embed_dims)

    # set pos
    pos = torch.rand(num_key, batch_size, embed_dims)
    out = module(x, pos)
    assert out.shape == (num_key, batch_size, embed_dims)

    # set key_padding_mask
    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(x, pos, None, key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)

    # set attn_mask
    attn_mask = torch.rand(num_key, num_key) > 0.5
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)

    # pre_norm
    order = ('norm', 'selfattn', 'norm', 'ffn')
    module = TransformerEncoder(
        num_layers, embed_dims, num_heads, feedforward_channels, order=order)
    assert module.pre_norm
    assert module.norm is not None
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, embed_dims)


def test_transformer_decoder(num_layers=2,
                             embed_dims=8,
                             num_heads=2,
                             feedforward_channels=8,
                             num_key=10,
                             num_query=5,
                             batch_size=1):
    module = TransformerDecoder(num_layers, embed_dims, num_heads,
                                feedforward_channels)
    query = torch.rand(num_query, batch_size, embed_dims)
    memory = torch.rand(num_key, batch_size, embed_dims)
    out = module(query, memory)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set query_pos
    query_pos = torch.rand(num_query, batch_size, embed_dims)
    out = module(query, memory, query_pos=query_pos)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set memory_pos
    memory_pos = torch.rand(num_key, batch_size, embed_dims)
    out = module(query, memory, memory_pos, query_pos)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set memory_key_padding_mask
    memory_key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set target_key_padding_mask
    target_key_padding_mask = torch.rand(batch_size, num_query) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask,
        target_key_padding_mask=target_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set memory_attn_mask
    memory_attn_mask = torch.rand(num_query, num_key) > 0.5
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask, None,
                 memory_key_padding_mask, target_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # set target_attn_mask
    target_attn_mask = torch.rand(num_query, num_query) > 0.5
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 target_attn_mask, memory_key_padding_mask,
                 target_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # pre_norm
    order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
    module = TransformerDecoder(
        num_layers, embed_dims, num_heads, feedforward_channels, order=order)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 target_attn_mask, memory_key_padding_mask,
                 target_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, embed_dims)

    # return_intermediate
    module = TransformerDecoder(
        num_layers,
        embed_dims,
        num_heads,
        feedforward_channels,
        order=order,
        return_intermediate=True)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 target_attn_mask, memory_key_padding_mask,
                 target_key_padding_mask)
    assert out.shape == (num_layers, num_query, batch_size, embed_dims)


def test_transformer(num_enc_layers=2,
                     num_dec_layers=2,
                     embed_dims=8,
                     num_heads=2,
                     num_query=5,
                     batch_size=1):
    module = Transformer(embed_dims, num_heads, num_enc_layers, num_dec_layers)
    height, width = 8, 6
    x = torch.rand(batch_size, embed_dims, height, width)
    mask = torch.rand(batch_size, height, width) > 0.5
    query_embed = torch.rand(num_query, embed_dims)
    pos_embed = torch.rand(batch_size, embed_dims, height, width)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (1, batch_size, num_query, embed_dims)
    assert mem.shape == (batch_size, embed_dims, height, width)

    # pre_norm
    module = Transformer(
        embed_dims, num_heads, num_enc_layers, num_dec_layers, pre_norm=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (1, batch_size, num_query, embed_dims)
    assert mem.shape == (batch_size, embed_dims, height, width)

    # return_intermediate
    module = Transformer(
        embed_dims,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        return_intermediate_dec=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (num_dec_layers, batch_size, num_query, embed_dims)
    assert mem.shape == (batch_size, embed_dims, height, width)

    # pre_norm and return_intermediate
    module = Transformer(
        embed_dims,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        pre_norm=True,
        return_intermediate_dec=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (num_dec_layers, batch_size, num_query, embed_dims)
    assert mem.shape == (batch_size, embed_dims, height, width)

    # test init_weights
    module.init_weights()
