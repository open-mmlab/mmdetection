import pytest
from mmcv.utils import ConfigDict

from mmdet.models.utils.transformer import (DetrTransformerDecoder,
                                            DetrTransformerEncoder,
                                            Transformer)


def test_detr_transformer_dencoder_encoder_layer():
    config = ConfigDict(
        dict(
            return_intermediate=True,
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=(
                    'norm',
                    'self_attn',
                    'norm',
                    'cross_attn',
                    'norm',
                    'ffn',
                ))))
    assert DetrTransformerDecoder(**config).layers[0].pre_norm
    assert len(DetrTransformerDecoder(**config).layers) == 6

    DetrTransformerDecoder(**config)
    with pytest.raises(AssertionError):
        config = ConfigDict(
            dict(
                return_intermediate=True,
                num_layers=6,
                transformerlayers=[
                    dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))
                ] * 5))
        DetrTransformerDecoder(**config)

    config = ConfigDict(
        dict(
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'))))

    with pytest.raises(AssertionError):
        # len(operation_order) == 6
        DetrTransformerEncoder(**config)


def test_transformer():
    config = ConfigDict(
        dict(
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )))
    transformer = Transformer(**config)
    transformer.init_weights()
