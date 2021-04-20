import pytest
import torch

from mmdet.models.backbones import RegNet

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


@pytest.mark.parametrize('arch_name,arch,out_channels', regnet_test_data)
def test_regnet_backbone(arch_name, arch, out_channels):
    with pytest.raises(AssertionError):
        # ResNeXt depth should be in [50, 101, 152]
        RegNet(arch_name + '233')

    # Test RegNet with arch_name
    model = RegNet(arch_name)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, out_channels[0], 56, 56])
    assert feat[1].shape == torch.Size([1, out_channels[1], 28, 28])
    assert feat[2].shape == torch.Size([1, out_channels[2], 14, 14])
    assert feat[3].shape == torch.Size([1, out_channels[3], 7, 7])

    # Test RegNet with arch
    model = RegNet(arch)
    assert feat[0].shape == torch.Size([1, out_channels[0], 56, 56])
    assert feat[1].shape == torch.Size([1, out_channels[1], 28, 28])
    assert feat[2].shape == torch.Size([1, out_channels[2], 14, 14])
    assert feat[3].shape == torch.Size([1, out_channels[3], 7, 7])
