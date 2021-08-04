import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones.csp_darknet import CSPDarknet
from .utils import check_norm_state, is_norm


def test_csp_darknet_backbone():
    with pytest.raises(ValueError):
        # frozen_stages must in range(-1, len(arch_setting) + 1)
        CSPDarknet(frozen_stages=6)

    with pytest.raises(AssertionError):
        # out_indices in range(len(arch_setting) + 1)
        CSPDarknet(out_indices=[6])

    # Test CSPDarknet with first stage frozen
    frozen_stages = 1
    model = CSPDarknet(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()

    for mod in model.stem.modules():
        for param in mod.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'stage{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test CSPDarknet with norm_eval=True
    model = CSPDarknet(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

    # Test CSPDarknet-P5 forward with widen_factor=1.0
    model = CSPDarknet(arch='P5', widen_factor=1.0, out_indices=range(0, 5))
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 64, 112, 112))
    assert feat[1].shape == torch.Size((1, 128, 56, 56))
    assert feat[2].shape == torch.Size((1, 256, 28, 28))
    assert feat[3].shape == torch.Size((1, 512, 14, 14))
    assert feat[4].shape == torch.Size((1, 1024, 7, 7))

    # Test CSPDarknet-P5 forward with widen_factor=0.5
    model = CSPDarknet(arch='P5', widen_factor=0.5, out_indices=range(0, 5))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 32, 112, 112))
    assert feat[1].shape == torch.Size((1, 64, 56, 56))
    assert feat[2].shape == torch.Size((1, 128, 28, 28))
    assert feat[3].shape == torch.Size((1, 256, 14, 14))
    assert feat[4].shape == torch.Size((1, 512, 7, 7))

    # Test CSPDarknet-P6 forward with widen_factor=1.5
    model = CSPDarknet(
        arch='P6',
        widen_factor=1.5,
        out_indices=range(0, 6),
        spp_kernal_sizes=(3, 5, 7))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 320, 320)
    feat = model(imgs)
    assert feat[0].shape == torch.Size((1, 96, 160, 160))
    assert feat[1].shape == torch.Size((1, 192, 80, 80))
    assert feat[2].shape == torch.Size((1, 384, 40, 40))
    assert feat[3].shape == torch.Size((1, 768, 20, 20))
    assert feat[4].shape == torch.Size((1, 1152, 10, 10))
    assert feat[5].shape == torch.Size((1, 1536, 5, 5))

    # Test CSPDarknet forward with dict(type='ReLU')
    model = CSPDarknet(
        widen_factor=1.0, act_cfg=dict(type='ReLU'), out_indices=range(0, 5))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 64, 112, 112))
    assert feat[1].shape == torch.Size((1, 128, 56, 56))
    assert feat[2].shape == torch.Size((1, 256, 28, 28))
    assert feat[3].shape == torch.Size((1, 512, 14, 14))
    assert feat[4].shape == torch.Size((1, 1024, 7, 7))

    # Test CSPDarknet with BatchNorm forward
    model = CSPDarknet(widen_factor=1.0, out_indices=range(0, 5))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 64, 112, 112))
    assert feat[1].shape == torch.Size((1, 128, 56, 56))
    assert feat[2].shape == torch.Size((1, 256, 28, 28))
    assert feat[3].shape == torch.Size((1, 512, 14, 14))
    assert feat[4].shape == torch.Size((1, 1024, 7, 7))

    # Test CSPDarknet with custom arch forward
    arch_ovewrite = [[32, 56, 3, True, False], [56, 224, 2, True, False],
                     [224, 512, 1, True, False]]
    model = CSPDarknet(
        arch_ovewrite=arch_ovewrite,
        widen_factor=1.0,
        out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size((1, 32, 112, 112))
    assert feat[1].shape == torch.Size((1, 56, 56, 56))
    assert feat[2].shape == torch.Size((1, 224, 28, 28))
    assert feat[3].shape == torch.Size((1, 512, 14, 14))
