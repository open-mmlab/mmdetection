# Copyright (c) OpenMMLab. All rights reserved.
import sys
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmengine.config import Config

# Add tools path so we can import the module directly
sys.path.insert(0, 'tools/analysis_tools')
_mod = __import__('get_flops_grounding')
_get_backbone_out_channels = _mod._get_backbone_out_channels
_get_feature_strides = _mod._get_feature_strides
_get_neck_out_channels = _mod._get_neck_out_channels
_get_transformer_config = _mod._get_transformer_config
count_neck_flops = _mod.count_neck_flops
count_text_encoder_flops = _mod.count_text_encoder_flops
count_transformer_flops = _mod.count_transformer_flops
format_flops = _mod.format_flops
format_params = _mod.format_params


class TestFormatFlops(unittest.TestCase):
    """Test format_flops helper."""

    def test_tera(self):
        self.assertEqual(format_flops(1.5e12), '1.50 T')

    def test_giga(self):
        self.assertEqual(format_flops(2.34e9), '2.34 G')

    def test_mega(self):
        self.assertEqual(format_flops(5.67e6), '5.67 M')

    def test_small(self):
        self.assertEqual(format_flops(1234.0), '1234.00')

    def test_exact_boundary_tera(self):
        self.assertEqual(format_flops(1e12), '1.00 T')

    def test_exact_boundary_giga(self):
        self.assertEqual(format_flops(1e9), '1.00 G')

    def test_exact_boundary_mega(self):
        self.assertEqual(format_flops(1e6), '1.00 M')


class TestFormatParams(unittest.TestCase):
    """Test format_params helper."""

    def test_billion(self):
        self.assertEqual(format_params(1.2e9), '1.20 B')

    def test_million(self):
        self.assertEqual(format_params(27.52e6), '27.52 M')

    def test_thousand(self):
        self.assertEqual(format_params(3.5e3), '3.50 K')

    def test_small(self):
        self.assertEqual(format_params(42), '42')

    def test_exact_boundary_billion(self):
        self.assertEqual(format_params(1e9), '1.00 B')

    def test_exact_boundary_million(self):
        self.assertEqual(format_params(1e6), '1.00 M')

    def test_exact_boundary_thousand(self):
        self.assertEqual(format_params(1e3), '1.00 K')


class TestGetBackboneOutChannels(unittest.TestCase):
    """Test _get_backbone_out_channels config reader."""

    def test_from_neck_in_channels(self):
        cfg = Config(
            dict(
                model=dict(
                    neck=dict(in_channels=[192, 384, 768]),
                    backbone=dict(type='SwinTransformer'))))
        result = _get_backbone_out_channels(cfg)
        self.assertEqual(result, [192, 384, 768])

    def test_fallback_swin(self):
        cfg = Config(
            dict(
                model=dict(
                    backbone=dict(
                        type='SwinTransformer',
                        embed_dims=96,
                        depths=[2, 2, 6, 2]))))
        result = _get_backbone_out_channels(cfg)
        self.assertEqual(result, [96, 192, 384, 768])

    def test_fallback_swin_custom_dims(self):
        cfg = Config(
            dict(
                model=dict(
                    backbone=dict(
                        type='SwinTransformer',
                        embed_dims=128,
                        depths=[2, 2, 18, 2]))))
        result = _get_backbone_out_channels(cfg)
        self.assertEqual(result, [128, 256, 512, 1024])

    def test_fallback_generic(self):
        cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
        result = _get_backbone_out_channels(cfg)
        self.assertEqual(result, [256, 512, 1024, 2048])


class TestGetFeatureStrides(unittest.TestCase):
    """Test _get_feature_strides config reader."""

    def test_from_neck(self):
        cfg = Config(dict(model=dict(neck=dict(in_channels=[192, 384, 768]))))
        result = _get_feature_strides(cfg)
        self.assertEqual(result, [8, 16, 32])

    def test_four_levels(self):
        cfg = Config(
            dict(model=dict(neck=dict(in_channels=[96, 192, 384, 768]))))
        result = _get_feature_strides(cfg)
        self.assertEqual(result, [8, 16, 32, 64])

    def test_fallback(self):
        cfg = Config(dict(model=dict()))
        result = _get_feature_strides(cfg)
        self.assertEqual(result, [8, 16, 32])


class TestGetNeckOutChannels(unittest.TestCase):
    """Test _get_neck_out_channels config reader."""

    def test_from_config(self):
        cfg = Config(dict(model=dict(neck=dict(out_channels=256))))
        self.assertEqual(_get_neck_out_channels(cfg), 256)

    def test_custom_channels(self):
        cfg = Config(dict(model=dict(neck=dict(out_channels=512))))
        self.assertEqual(_get_neck_out_channels(cfg), 512)

    def test_fallback(self):
        cfg = Config(dict(model=dict()))
        self.assertEqual(_get_neck_out_channels(cfg), 256)


class TestGetTransformerConfig(unittest.TestCase):
    """Test _get_transformer_config config reader."""

    def test_defaults(self):
        cfg = Config(dict(model=dict()))
        result = _get_transformer_config(cfg)
        self.assertEqual(result['embed_dim'], 256)
        self.assertEqual(result['num_encoder_layers'], 6)
        self.assertEqual(result['num_decoder_layers'], 6)
        self.assertEqual(result['ffn_dim'], 2048)
        self.assertEqual(result['num_queries'], 900)

    def test_from_encoder_config(self):
        cfg = Config(
            dict(
                model=dict(
                    encoder=dict(
                        num_layers=3,
                        layer_cfg=dict(
                            self_attn_cfg=dict(embed_dims=512),
                            ffn_cfg=dict(feedforward_channels=1024))))))
        result = _get_transformer_config(cfg)
        self.assertEqual(result['num_encoder_layers'], 3)
        self.assertEqual(result['embed_dim'], 512)
        self.assertEqual(result['ffn_dim'], 1024)

    def test_from_decoder_config(self):
        cfg = Config(dict(model=dict(decoder=dict(num_layers=4))))
        result = _get_transformer_config(cfg)
        self.assertEqual(result['num_decoder_layers'], 4)

    def test_num_queries_from_model(self):
        cfg = Config(dict(model=dict(num_queries=300)))
        result = _get_transformer_config(cfg)
        self.assertEqual(result['num_queries'], 300)

    def test_num_queries_from_bbox_head(self):
        cfg = Config(dict(model=dict(bbox_head=dict(num_queries=100))))
        result = _get_transformer_config(cfg)
        self.assertEqual(result['num_queries'], 100)


class TestCountTextEncoderFlops(unittest.TestCase):
    """Test count_text_encoder_flops."""

    def _make_model_with_lang(self, name):
        """Create a mock model with a language_model attribute."""
        model = MagicMock()
        lang = MagicMock()
        lang.name = name
        # Create a small parameter for counting
        param = nn.Parameter(torch.randn(10, 10))
        lang.parameters.return_value = [param]
        model.language_model = lang
        return model

    def test_no_language_model(self):
        model = MagicMock(spec=[])  # no language_model attr
        flops, params = count_text_encoder_flops(model, 80)
        self.assertIsNone(flops)
        self.assertIsNone(params)

    def test_clip_base(self):
        model = self._make_model_with_lang('clip-base')
        flops, params = count_text_encoder_flops(model, 80)
        self.assertEqual(flops, 4e9 * 80)
        self.assertEqual(params, 100)

    def test_clip_large(self):
        model = self._make_model_with_lang('clip-large')
        flops, params = count_text_encoder_flops(model, 1)
        self.assertEqual(flops, 10e9)

    def test_bert(self):
        model = self._make_model_with_lang('bert-base')
        flops, params = count_text_encoder_flops(model, 1)
        self.assertEqual(flops, 11e9)

    def test_unknown_defaults(self):
        model = self._make_model_with_lang('some-model')
        flops, params = count_text_encoder_flops(model, 1)
        self.assertEqual(flops, 5e9)

    def test_num_classes_none(self):
        model = self._make_model_with_lang('clip-base')
        flops, _ = count_text_encoder_flops(model, None)
        self.assertEqual(flops, 4e9)


class TestCountNeckFlops(unittest.TestCase):
    """Test count_neck_flops."""

    def test_no_neck(self):
        model = MagicMock(spec=[])
        cfg = Config(dict(model=dict()))
        flops, params = count_neck_flops(model, (800, 1333), cfg)
        self.assertEqual(flops, 0)
        self.assertEqual(params, 0)

    def test_neck_none(self):
        model = MagicMock()
        model.neck = None
        cfg = Config(dict(model=dict()))
        flops, params = count_neck_flops(model, (800, 1333), cfg)
        self.assertEqual(flops, 0)
        self.assertEqual(params, 0)

    def test_with_neck(self):
        model = MagicMock()
        neck = nn.Conv2d(3, 3, 1)
        model.neck = neck
        cfg = Config(
            dict(
                model=dict(
                    neck=dict(in_channels=[192, 384, 768], out_channels=256),
                    backbone=dict(type='SwinTransformer'))))
        flops, params = count_neck_flops(model, (800, 1333), cfg)
        self.assertGreater(flops, 0)
        self.assertGreater(params, 0)


class TestCountTransformerFlops(unittest.TestCase):
    """Test count_transformer_flops."""

    def test_basic(self):
        model = MagicMock(spec=[])
        cfg = Config(dict(model=dict(neck=dict(in_channels=[192, 384, 768]))))
        enc_f, dec_f, enc_p, dec_p = count_transformer_flops(
            model, (800, 1333), cfg)
        self.assertGreater(enc_f, 0)
        self.assertGreater(dec_f, 0)
        # No encoder/decoder attrs on mock, so params = 0
        self.assertEqual(enc_p, 0)
        self.assertEqual(dec_p, 0)

    def test_with_encoder_decoder(self):
        model = MagicMock()
        enc_param = nn.Parameter(torch.randn(10, 10))
        model.encoder.parameters.return_value = [enc_param]
        dec_param = nn.Parameter(torch.randn(5, 5))
        model.decoder.parameters.return_value = [dec_param]

        cfg = Config(
            dict(
                model=dict(
                    encoder=dict(num_layers=6),
                    decoder=dict(num_layers=6),
                    neck=dict(in_channels=[192, 384, 768]))))
        enc_f, dec_f, enc_p, dec_p = count_transformer_flops(
            model, (800, 1333), cfg)
        self.assertGreater(enc_f, 0)
        self.assertGreater(dec_f, 0)
        self.assertEqual(enc_p, 100)  # 10*10
        self.assertEqual(dec_p, 25)  # 5*5

    def test_custom_config(self):
        """Verify different configs produce different FLOPs."""
        model = MagicMock(spec=[])
        cfg_small = Config(
            dict(
                model=dict(
                    encoder=dict(num_layers=3),
                    decoder=dict(num_layers=3),
                    neck=dict(in_channels=[192, 384, 768]))))
        cfg_large = Config(
            dict(
                model=dict(
                    encoder=dict(num_layers=6),
                    decoder=dict(num_layers=6),
                    neck=dict(in_channels=[192, 384, 768]))))

        enc_small, dec_small, _, _ = count_transformer_flops(
            model, (800, 1333), cfg_small)
        enc_large, dec_large, _, _ = count_transformer_flops(
            model, (800, 1333), cfg_large)

        self.assertGreater(enc_large, enc_small)
        self.assertGreater(dec_large, dec_small)


if __name__ == '__main__':
    unittest.main()
