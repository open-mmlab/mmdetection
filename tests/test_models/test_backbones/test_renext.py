import pytest
import torch

from mmdet.models.backbones import ResNeXt
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from .utils import is_block


def test_renext_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckX(64, 64, groups=32, base_width=4, style='tensorflow')

    # Test ResNeXt Bottleneck structure
    block = BottleneckX(
        64, 64, groups=32, base_width=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    # Test ResNeXt Bottleneck with DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        # conv_cfg must be None if dcn is not None
        BottleneckX(
            64,
            64,
            groups=32,
            base_width=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    BottleneckX(64, 64, dcn=dcn)

    # Test ResNeXt Bottleneck forward
    block = BottleneckX(64, 16, groups=32, base_width=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test ResNeXt Bottleneck forward with plugins
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            stages=(False, False, True, True),
            position='after_conv2')
    ]
    block = BottleneckX(64, 16, groups=32, base_width=4, plugins=plugins)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnext_backbone():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

    # Test ResNeXt with group 32, base_width 4
    model = ResNeXt(depth=50, groups=32, base_width=4)
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])


regnet_test_data = [
    ('regnetx_400mf',
     dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22,
          bot_mul=1.0), [32, 64, 160, 384]),
    ('regnetx_800mf',
     dict(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16,
          bot_mul=1.0), [64, 128, 288, 672]),
    ('regnetx_1.6gf',
     dict(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18,
          bot_mul=1.0), [72, 168, 408, 912]),
    ('regnetx_3.2gf',
     dict(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25,
          bot_mul=1.0), [96, 192, 432, 1008]),
    ('regnetx_4.0gf',
     dict(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23,
          bot_mul=1.0), [80, 240, 560, 1360]),
    ('regnetx_6.4gf',
     dict(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17,
          bot_mul=1.0), [168, 392, 784, 1624]),
    ('regnetx_8.0gf',
     dict(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23,
          bot_mul=1.0), [80, 240, 720, 1920]),
    ('regnetx_12gf',
     dict(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19,
          bot_mul=1.0), [224, 448, 896, 2240]),
]
