import pytest
import torch

from mmdet.models.backbones.pvt import (PVTEncoderLayer,
                                        PyramidVisionTransformer,
                                        PyramidVisionTransformerV2)


def test_pvt_block():
    # test PVT structure and forward
    block = PVTEncoderLayer(
        embed_dims=64, num_heads=4, feedforward_channels=256)
    assert block.ffn.embed_dims == 64
    assert block.attn.num_heads == 4
    assert block.ffn.feedforward_channels == 256
    x = torch.randn(1, 56 * 56, 64)
    x_out = block(x, (56, 56))
    assert x_out.shape == torch.Size([1, 56 * 56, 64])


def test_pvt():
    """Test PVT backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        PyramidVisionTransformer(pretrained=123)

    # test pretrained image size
    with pytest.raises(AssertionError):
        PyramidVisionTransformer(pretrain_img_size=(224, 224, 224))

    # Test absolute position embedding
    temp = torch.randn((1, 3, 224, 224))
    model = PyramidVisionTransformer(
        pretrain_img_size=224, use_abs_pos_embed=True)
    model.init_weights()
    model(temp)

    # Test normal inference
    temp = torch.randn((1, 3, 512, 512))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 128, 128)
    assert outs[1].shape == (1, 128, 64, 64)
    assert outs[2].shape == (1, 320, 32, 32)
    assert outs[3].shape == (1, 512, 16, 16)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 511, 511))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 127, 127)
    assert outs[1].shape == (1, 128, 63, 63)
    assert outs[2].shape == (1, 320, 31, 31)
    assert outs[3].shape == (1, 512, 15, 15)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 28, 34)
    assert outs[1].shape == (1, 128, 14, 17)
    assert outs[2].shape == (1, 320, 7, 8)
    assert outs[3].shape == (1, 512, 3, 4)


def test_pvtv2():
    """Test PVTv2 backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        PyramidVisionTransformerV2(pretrained=123)

    # test pretrained image size
    with pytest.raises(AssertionError):
        PyramidVisionTransformerV2(pretrain_img_size=(224, 224, 224))

    # Test normal inference
    temp = torch.randn((1, 3, 512, 512))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 128, 128)
    assert outs[1].shape == (1, 128, 64, 64)
    assert outs[2].shape == (1, 320, 32, 32)
    assert outs[3].shape == (1, 512, 16, 16)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 511, 511))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 128, 128)
    assert outs[1].shape == (1, 128, 64, 64)
    assert outs[2].shape == (1, 320, 32, 32)
    assert outs[3].shape == (1, 512, 16, 16)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 28, 35)
    assert outs[1].shape == (1, 128, 14, 18)
    assert outs[2].shape == (1, 320, 7, 9)
    assert outs[3].shape == (1, 512, 4, 5)
