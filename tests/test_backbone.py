import pytest
import torch

from mmdet.models.backbones import ResNet, ResNetV1d, ResNeXt


def test_basic_block():
    inputs = torch.rand(2, 3, 224, 224)
    self = ResNet(depth=18)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_bottleneck_block():
    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(depth=50)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    self = ResNeXt(depth=50, groups=32, base_width=4)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_caffe_style():
    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(depth=50, style='caffe')
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    self = ResNeXt(depth=50, groups=32, base_width=4, style='caffe')
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_init_weights():
    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(depth=50)
    self.init_weights()
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    with pytest.raises(TypeError):
        self.init_weights(2333)


def test_frozen_stages():

    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(depth=50, frozen_stages=1)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_resnet_wrong_depth():
    with pytest.raises(KeyError):
        ResNet(depth=233)


def test_resnetv1d():
    inputs = torch.rand(2, 3, 224, 224)
    self = ResNetV1d(depth=50)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_dcn_build():
    stage_with_dcn = (False, True, True, True)
    dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
    ResNet(depth=50, dcn=dcn, stage_with_dcn=stage_with_dcn)
    ResNeXt(
        depth=50,
        groups=32,
        base_width=4,
        dcn=dcn,
        stage_with_dcn=stage_with_dcn)


def test_gcb():
    gcb = dict(ratio=1. / 16., )
    stage_with_gcb = (False, True, True, True)
    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(depth=50, gcb=gcb, stage_with_gcb=stage_with_gcb)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    self = ResNeXt(
        depth=50,
        groups=32,
        base_width=4,
        gcb=gcb,
        stage_with_gcb=stage_with_gcb)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def test_gen_attention():
    gen_attention = dict(
        spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2)
    stage_with_gen_attention = [[], [], [0, 1, 2, 3, 4, 5], [0, 1, 2]]
    inputs = torch.rand(2, 3, 224, 224)

    self = ResNet(
        depth=50,
        gen_attention=gen_attention,
        stage_with_gen_attention=stage_with_gen_attention)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    self = ResNeXt(
        depth=50,
        gen_attention=gen_attention,
        stage_with_gen_attention=stage_with_gen_attention)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
