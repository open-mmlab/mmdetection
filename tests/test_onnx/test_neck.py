import numpy as np
import torch

from mmdet.models.necks import FPN, YOLOV3Neck
from .utils import get_data_path, list_gen, verify_model

onnx_io = 'tmp.onnx'

# Control the returned model of neck_config()
test_step_names = {
    'normal': 0,
    'wo_extra_convs': 1,
    'lateral_bns': 2,
    'bilinear_upsample': 3,
    'scale_factor': 4,
    'extra_convs_inputs': 5,
    'extra_convs_laterals': 6,
    'extra_convs_outputs': 7,
}

data_path = get_data_path()


def ort_validate(fpn_model, feats):
    with torch.no_grad():
        torch.onnx.export(
            fpn_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = fpn_model.forward(feats)
    torch_outputs = list_gen(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def fpn_config(test_step_name):
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    if (test_step_names[test_step_name] == 0):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 1):
        # Tests for fpn with no extra convs (pooling is used instead)
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=False,
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 2):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            no_norm_on_lateral=False,
            norm_cfg=dict(type='BN', requires_grad=True),
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 3):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            upsample_cfg=dict(mode='bilinear', align_corners=True),
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 4):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            upsample_cfg=dict(scale_factor=2),
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 5):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs='on_input',
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 6):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs='on_lateral',
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats
    elif (test_step_names[test_step_name] == 7):
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs='on_output',
            num_outs=5)
        fpn_model.cpu().eval()
        return fpn_model, feats


def yolo_config(test_step_name):
    in_channels = [16, 8, 4]
    out_channels = [8, 4, 2]

    # FPN expects a multiple levels of features per image
    feats = []
    yolov3_neck_data = 'yolov3_neck_'
    for i in range(len(in_channels)):
        data_name = data_path + yolov3_neck_data + str(i) + '.npy'
        feats.append(torch.tensor(np.load(data_name)))

    if (test_step_names[test_step_name] == 0):
        yolo_model = YOLOV3Neck(
            in_channels=in_channels, out_channels=out_channels, num_scales=3)
        yolo_model.cpu().eval()
        return yolo_model, feats


def test_fpn_normal():
    outs = fpn_config('normal')
    ort_validate(*outs)


def test_fpn_wo_extra_convs():
    outs = fpn_config('wo_extra_convs')
    ort_validate(*outs)


def test_fpn_lateral_bns():
    outs = fpn_config('lateral_bns')
    ort_validate(*outs)


def test_fpn_bilinear_upsample():
    outs = fpn_config('bilinear_upsample')
    ort_validate(*outs)


def test_fpn_scale_factor():
    outs = fpn_config('scale_factor')
    ort_validate(*outs)


def test_fpn_extra_convs_inputs():
    outs = fpn_config('extra_convs_inputs')
    ort_validate(*outs)


def test_fpn_extra_convs_laterals():
    outs = fpn_config('extra_convs_laterals')
    ort_validate(*outs)


def test_fpn_extra_convs_outputs():
    outs = fpn_config('extra_convs_outputs')
    ort_validate(*outs)


def test_yolo_normal():
    outs = yolo_config('normal')
    ort_validate(*outs)
