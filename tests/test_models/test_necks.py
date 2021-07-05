import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.necks import (FPN, ChannelMapper, CTResNetNeck,
                                DilatedEncoder, SSDNeck, YOLOV3Neck)


def test_fpn():
    """Tests fpn."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1)

    # Invalid `add_extra_convs` option
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs='on_xxx',
            num_outs=5)

    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        num_outs=5)

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = fpn_model(feats)
    assert fpn_model.add_extra_convs == 'on_input'
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Tests for fpn with no extra convs (pooling is used instead)
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=False,
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert not fpn_model.add_extra_convs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Tests for fpn with lateral bns
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        no_norm_on_lateral=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
    bn_exist = False
    for m in fpn_model.modules():
        if isinstance(m, _BatchNorm):
            bn_exist = True
    assert bn_exist

    # Bilinear upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        num_outs=5)
    fpn_model(feats)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Scale factor instead of fixed upsample size upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(scale_factor=2),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'inputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_input',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_input'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'laterals'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_lateral',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_lateral'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'outputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_output'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_channel_mapper():
    """Tests ChannelMapper."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    kernel_size = 3
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    # in_channels must be a list
    with pytest.raises(AssertionError):
        channel_mapper = ChannelMapper(
            in_channels=10, out_channels=out_channels, kernel_size=kernel_size)
    # the length of channel_mapper's inputs must be equal to the length of
    # in_channels
    with pytest.raises(AssertionError):
        channel_mapper = ChannelMapper(
            in_channels=in_channels[:-1],
            out_channels=out_channels,
            kernel_size=kernel_size)
        channel_mapper(feats)

    channel_mapper = ChannelMapper(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size)

    outs = channel_mapper(feats)
    assert len(outs) == len(feats)
    for i in range(len(feats)):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_dilated_encoder():
    in_channels = 16
    out_channels = 32
    out_shape = 34
    dilated_encoder = DilatedEncoder(in_channels, out_channels, 16, 2)
    feat = [torch.rand(1, in_channels, 34, 34)]
    out_feat = dilated_encoder(feat)[0]
    assert out_feat.shape == (1, out_channels, out_shape, out_shape)


def test_ct_resnet_neck():
    # num_filters/num_kernels must be a list
    with pytest.raises(TypeError):
        CTResNetNeck(
            in_channel=10, num_deconv_filters=10, num_deconv_kernels=4)

    # num_filters/num_kernels must be same length
    with pytest.raises(AssertionError):
        CTResNetNeck(
            in_channel=10,
            num_deconv_filters=(10, 10),
            num_deconv_kernels=(4, ))

    in_channels = 16
    num_filters = (8, 8)
    num_kernels = (4, 4)
    feat = torch.rand(1, 16, 4, 4)
    ct_resnet_neck = CTResNetNeck(
        in_channel=in_channels,
        num_deconv_filters=num_filters,
        num_deconv_kernels=num_kernels,
        use_dcn=False)

    # feat must be list or tuple
    with pytest.raises(AssertionError):
        ct_resnet_neck(feat)

    out_feat = ct_resnet_neck([feat])[0]
    assert out_feat.shape == (1, num_filters[-1], 16, 16)

    if torch.cuda.is_available():
        # test dcn
        ct_resnet_neck = CTResNetNeck(
            in_channel=in_channels,
            num_deconv_filters=num_filters,
            num_deconv_kernels=num_kernels)
        ct_resnet_neck = ct_resnet_neck.cuda()
        feat = feat.cuda()
        out_feat = ct_resnet_neck([feat])[0]
        assert out_feat.shape == (1, num_filters[-1], 16, 16)


def test_yolov3_neck():
    # num_scales, in_channels, out_channels must be same length
    with pytest.raises(AssertionError):
        YOLOV3Neck(num_scales=3, in_channels=[16, 8, 4], out_channels=[8, 4])

    # len(feats) must equal to num_scales
    with pytest.raises(AssertionError):
        neck = YOLOV3Neck(
            num_scales=3, in_channels=[16, 8, 4], out_channels=[8, 4, 2])
        feats = (torch.rand(1, 4, 16, 16), torch.rand(1, 8, 16, 16))
        neck(feats)

    # test normal channels
    s = 32
    in_channels = [16, 8, 4]
    out_channels = [8, 4, 2]
    feat_sizes = [s // 2**i for i in range(len(in_channels) - 1, -1, -1)]
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels) - 1, -1, -1)
    ]
    neck = YOLOV3Neck(
        num_scales=3, in_channels=in_channels, out_channels=out_channels)
    outs = neck(feats)

    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape == \
               (1, out_channels[i], feat_sizes[i], feat_sizes[i])

    # test more flexible setting
    s = 32
    in_channels = [32, 8, 16]
    out_channels = [19, 21, 5]
    feat_sizes = [s // 2**i for i in range(len(in_channels) - 1, -1, -1)]
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels) - 1, -1, -1)
    ]
    neck = YOLOV3Neck(
        num_scales=3, in_channels=in_channels, out_channels=out_channels)
    outs = neck(feats)

    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape == \
               (1, out_channels[i], feat_sizes[i], feat_sizes[i])


def test_ssd_neck():
    # level_strides/level_paddings must be same length
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[8, 16, 32],
            level_strides=[2],
            level_paddings=[2, 1])

    # length of out_channels must larger than in_channels
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[8],
            level_strides=[2],
            level_paddings=[2])

    # len(out_channels) - len(in_channels) must equal to len(level_strides)
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[4, 16, 64],
            level_strides=[2, 2],
            level_paddings=[2, 2])

    # in_channels must be same with out_channels[:len(in_channels)]
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[4, 16, 64],
            level_strides=[2],
            level_paddings=[2])

    ssd_neck = SSDNeck(
        in_channels=[4],
        out_channels=[4, 8, 16],
        level_strides=[2, 1],
        level_paddings=[1, 0])
    feats = (torch.rand(1, 4, 16, 16), )
    outs = ssd_neck(feats)
    assert outs[0].shape == (1, 4, 16, 16)
    assert outs[1].shape == (1, 8, 8, 8)
    assert outs[2].shape == (1, 16, 6, 6)

    # test SSD-Lite Neck
    ssd_neck = SSDNeck(
        in_channels=[4, 8],
        out_channels=[4, 8, 16],
        level_strides=[1],
        level_paddings=[1],
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU6'))
    assert not hasattr(ssd_neck, 'l2_norm')

    from mmcv.cnn.bricks import DepthwiseSeparableConvModule
    assert isinstance(ssd_neck.extra_layers[0][-1],
                      DepthwiseSeparableConvModule)

    feats = (torch.rand(1, 4, 8, 8), torch.rand(1, 8, 8, 8))
    outs = ssd_neck(feats)
    assert outs[0].shape == (1, 4, 8, 8)
    assert outs[1].shape == (1, 8, 8, 8)
    assert outs[2].shape == (1, 16, 8, 8)
