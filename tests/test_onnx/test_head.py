import os.path as osp
import pickle
from functools import partial

import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.cnn import Scale

from mmdet import digit_version
from mmdet.models.dense_heads import (FCOSHead, FSAFHead, RetinaHead, SSDHead,
                                      YOLOV3Head)
from .utils import flat, verify_model

onnx_io = 'tmp.onnx'

data_path = osp.join(osp.dirname(__file__), 'data/')

min_required_version = '1.5.0'
if digit_version(torch.__version__) <= digit_version(min_required_version):
    pytest.skip(
        'ort backend does not support version below 1.5.0',
        allow_module_level=True)


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def retinanet_config():
    """RetinanNet Head Config."""

    head_cfg = dict(
        stacked_convs=6,
        feat_channels=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]))

    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = RetinaHead(
        num_classes=4, in_channels=1, test_cfg=test_cfg, **head_cfg)
    model.requires_grad_(False)
    model.eval()

    return model


def test_retina_head_forward_single():
    """Test RetinaNet Head single forward in torch and onnxruntime env."""
    retina_model = retinanet_config()

    feat = torch.rand(1, retina_model.in_channels, 32, 32)
    wrap_model = WrapFunction(retina_model.forward_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feat,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feat)

    torch_outputs = wrap_model.forward(feat)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_retina_head_forward():
    """Test RetinaNet Head forward in torch and onnxruntime env."""
    retina_model = retinanet_config()
    s = 128
    feats = [
        torch.rand(1, retina_model.in_channels, s // (2**(i + 2)),
                   s // (2**(i + 2)))
        for i in range(len(retina_model.anchor_generator.strides))
    ]

    wrap_model = WrapFunction(retina_model.forward)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = wrap_model.forward(feats)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_retinanet_head_get_bboxes_single():
    retina_model = retinanet_config()
    s = 128

    retina_head_data = 'retina_head_get_bboxes.pkl'
    with open(data_path + retina_head_data, 'rb') as f:
        feats = pickle.load(f)
    feats = [feat[0] for feat in feats]
    cls_score = feats[:5]
    bboxes = feats[5:]

    featmap_sizes = [
        cls_score[i][0].shape[-2:]
        for i in range(len(retina_model.anchor_generator.strides))
    ]

    anchors = retina_model.anchor_generator.grid_anchors(
        featmap_sizes, device='cpu')

    retina_model._get_bboxes_single = partial(
        retina_model._get_bboxes_single,
        img_shape=(s, s, 3),
        scale_factor=1,
        cfg=None)
    wrap_model = WrapFunction(retina_model._get_bboxes_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes, anchors),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes + anchors)

    torch_outputs = wrap_model.forward(cls_score, bboxes, anchors)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_retinanet_head_get_bboxes():
    retina_model = retinanet_config()
    s = 128
    img_metas = [{
        'img_shape_for_onnx': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    retina_head_data = 'retina_head_get_bboxes.pkl'
    with open(data_path + retina_head_data, 'rb') as f:
        feats = pickle.load(f)
    cls_score = feats[:5]
    bboxes = feats[5:]

    retina_model.get_bboxes = partial(
        retina_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(retina_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes)

    torch_outputs = wrap_model.forward(cls_score, bboxes)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def yolo_config():
    """YoloV3 Head Config."""

    head_cfg = dict(
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'))

    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100))

    model = YOLOV3Head(
        num_classes=4,
        in_channels=[1, 1, 1],
        out_channels=[16, 8, 4],
        test_cfg=test_cfg,
        **head_cfg)
    model.requires_grad_(False)
    model.eval()

    return model


def test_yolov3_head_forward():
    yolo_model = yolo_config()

    feats = [
        torch.rand(1, 1, 64 // (2**(i + 2)), 64 // (2**(i + 2)))
        for i in range(len(yolo_model.in_channels))
    ]

    wrap_model = WrapFunction(yolo_model.forward)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = wrap_model.forward(feats)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_yolov3_head_get_bboxes_single():
    yolo_model = yolo_config()

    yolo_head_data = 'yolov3_head_get_bboxes.pkl'
    with open(data_path + yolo_head_data, 'rb') as f:
        pred_maps = pickle.load(f)
    pred_map = [pred_map[0] for pred_map in pred_maps]

    yolo_model._get_bboxes_single = partial(
        yolo_model._get_bboxes_single, scale_factor=1, cfg=None)
    wrap_model = WrapFunction(yolo_model._get_bboxes_single)
    wrap_model.cpu().eval()

    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            pred_map,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(pred_map)

    torch_outputs = wrap_model.forward(pred_map)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_yolov3_head_get_bboxes():
    yolo_model = yolo_config()

    s = 128
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    yolo_head_data = 'yolov3_head_get_bboxes.pkl'
    with open(data_path + yolo_head_data, 'rb') as f:
        pred_maps = pickle.load(f)

    yolo_model.get_bboxes = partial(yolo_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(yolo_model.get_bboxes)
    wrap_model.cpu().eval()

    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            pred_maps,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(pred_maps)

    torch_outputs = wrap_model.forward(pred_maps)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def fcos_config():
    """FCOS Head Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = FCOSHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    model.eval()
    return model


def test_fcos_head_forward_single():
    """Test fcos forward single in torch and ort env."""
    fcos_model = fcos_config()

    feat = torch.rand(1, fcos_model.in_channels, 32, 32)
    fcos_model.forward_single = partial(
        fcos_model.forward_single,
        scale=Scale(1.0).requires_grad_(False),
        stride=(4, ))
    wrap_model = WrapFunction(fcos_model.forward_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feat,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feat)

    torch_outputs = wrap_model.forward(feat)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fcos_head_forward():
    """Test fcos forward in mutil-level feature map."""
    fcos_model = fcos_config()

    s = 128
    feats = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]

    wrap_model = WrapFunction(fcos_model.forward)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = wrap_model.forward(feats)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fcos_head_get_bboxes_single():
    """Test fcos head get_bboxes_single() in ort and torch env."""
    fcos_model = fcos_config()

    s = 128
    cls_score = [
        torch.rand(fcos_model.num_classes, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    bbox = [
        torch.rand(4, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    centerness = [
        torch.rand(1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_score]
    fcos_model._get_bboxes_single = partial(
        fcos_model._get_bboxes_single,
        mlvl_points=fcos_model.get_points(featmap_sizes, torch.float32, 'cpu'),
        img_shape=(s, s, 3),
        scale_factor=1,
        cfg=None)

    wrap_model = WrapFunction(fcos_model._get_bboxes_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bbox, centerness),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bbox + centerness)

    torch_outputs = wrap_model.forward(cls_score, bbox, centerness)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fcos_head_get_bboxes():
    """Test fcos head get_bboxes() in ort and."""
    fcos_model = fcos_config()

    s = 128
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cls_scores = [
        torch.rand(1, fcos_model.num_classes, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    bboxes = [
        torch.rand(1, 4, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    centerness = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]

    fcos_model.get_bboxes = partial(fcos_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(fcos_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_scores, bboxes, centerness),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_scores + bboxes + centerness)

    torch_outputs = wrap_model.forward(cls_scores, bboxes, centerness)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def fsaf_config():
    cfg = dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]))

    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = FSAFHead(num_classes=4, in_channels=1, test_cfg=test_cfg, **cfg)
    model.requires_grad_(False)
    return model


def test_fsaf_head_forward_single():
    fsaf_model = fsaf_config()

    feat = torch.rand(1, fsaf_model.in_channels, 32, 32)
    wrap_model = WrapFunction(fsaf_model.forward_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feat,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feat)

    torch_outputs = wrap_model.forward(feat)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fsaf_head_forward():
    """Test RetinaNet Head forward in torch and onnxruntime env."""
    fsaf_model = fsaf_config()
    s = 128
    feats = [
        torch.rand(1, fsaf_model.in_channels, s // (2**(i + 2)),
                   s // (2**(i + 2)))
        for i in range(len(fsaf_model.anchor_generator.strides))
    ]

    wrap_model = WrapFunction(fsaf_model.forward)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = wrap_model.forward(feats)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fsaf_head_get_bboxes_single():
    fsaf_model = fsaf_config()
    s = 256

    fsaf_head_data = 'fsaf_head_get_bboxes.pkl'
    with open(data_path + fsaf_head_data, 'rb') as f:
        feats = pickle.load(f)
    feats = [feat[0] for feat in feats]
    cls_score = feats[:5]
    bbox = feats[5:]

    featmap_sizes = [
        cls_score[i].shape[-2:]
        for i in range(len(fsaf_model.anchor_generator.strides))
    ]

    anchors = fsaf_model.anchor_generator.grid_anchors(
        featmap_sizes, device='cpu')

    fsaf_model._get_bboxes_single = partial(
        fsaf_model._get_bboxes_single,
        mlvl_anchors=anchors,
        img_shape=(s, s, 3),
        scale_factor=1,
        cfg=None)
    wrap_model = WrapFunction(fsaf_model._get_bboxes_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bbox),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bbox)

    torch_outputs = flat(wrap_model.forward(cls_score, bbox))
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_fsaf_head_get_bboxes():
    fsaf_model = fsaf_config()
    s = 256
    img_metas = [{
        'img_shape_for_onnx': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    fsaf_head_data = 'fsaf_head_get_bboxes.pkl'
    with open(data_path + fsaf_head_data, 'rb') as f:
        feats = pickle.load(f)
    feats = [feat for feat in feats]
    cls_score = feats[:5]
    bboxes = feats[5:]

    fsaf_model.get_bboxes = partial(fsaf_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(fsaf_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes)

    torch_outputs = wrap_model.forward(cls_score, bboxes)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def ssd_config():
    """SSD Head Config."""
    cfg = dict(
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]))

    test_cfg = mmcv.Config(
        dict(
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200))

    model = SSDHead(
        num_classes=4,
        in_channels=(4, 8, 4, 2, 2, 2),
        test_cfg=test_cfg,
        **cfg)

    model.requires_grad_(False)
    model.eval()
    return model


def test_ssd_head_forward():
    ssd_model = ssd_config()

    featmap_size = [38, 19, 10, 6, 5, 3, 1]

    feats = [
        torch.rand(1, ssd_model.in_channels[i], featmap_size[i],
                   featmap_size[i]) for i in range(len(ssd_model.in_channels))
    ]

    wrap_model = WrapFunction(ssd_model.forward)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(feats)

    torch_outputs = wrap_model.forward(feats)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_ssd_head_get_bboxes_single():
    ssd_model = ssd_config()
    s = 300

    featmap_sizes = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]

    ssd_head_data = 'ssd_head_get_bboxes.pkl'
    with open(data_path + ssd_head_data, 'rb') as f:
        feats = pickle.load(f)
    feats = [feat[0] for feat in feats]
    cls_score = feats[:6]
    bboxes = feats[6:]

    anchors = ssd_model.anchor_generator.grid_anchors(
        featmap_sizes, device='cpu')

    ssd_model._get_bboxes_single = partial(
        ssd_model._get_bboxes_single,
        mlvl_anchors=anchors,
        img_shape=(s, s, 3),
        scale_factor=1,
        cfg=None)
    wrap_model = WrapFunction(ssd_model._get_bboxes_single)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes)

    torch_outputs = wrap_model.forward(cls_score, bboxes)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def test_ssd_head_get_bboxes():
    ssd_model = ssd_config()
    s = 300
    img_metas = [{
        'img_shape_for_onnx': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    ssd_head_data = 'ssd_head_get_bboxes.pkl'
    with open(data_path + ssd_head_data, 'rb') as f:
        feats = pickle.load(f)
    cls_score = feats[:6]
    bboxes = feats[6:]

    ssd_model.get_bboxes = partial(ssd_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(ssd_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes),
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes)

    torch_outputs = wrap_model.forward(cls_score, bboxes)
    torch_outputs = flat(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)
