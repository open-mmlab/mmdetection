# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import os.path as osp
import sys
from copy import copy
from subprocess import call, check_call, CalledProcessError, DEVNULL

import mmcv
import numpy as np
import onnx
import torch
import torch.onnx.symbolic_helper as sym_help
from mmcv.parallel import collate, scatter

from mmdet.apis import init_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmdet.models import detectors
from mmdet.models.anchor_heads.anchor_head import AnchorHead
from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.ops import RoIAlign
from mmdet.utils.deployment import register_extra_symbolics, TracerStub
from mmdet.utils.deployment.ssd_export_helpers import *


def export_to_onnx(model,
                   data,
                   export_name,
                   verbose=False,
                   strip_doc_string=False,
                   opset=10,
                   alt_ssd_export=False):
    register_extra_symbolics(opset)

    if alt_ssd_export:
        assert isinstance(model, detectors.SingleStageDetector)
        
        model.onnx_export = onnx_export.__get__(model)
        model.forward = forward.__get__(model)
        model.forward_export = forward_export_detector.__get__(model)
        model.bbox_head.export_forward = export_forward_ssd_head.__get__(model.bbox_head)
        model.bbox_head._prepare_cls_scores_bbox_preds = prepare_cls_scores_bbox_preds_ssd_head.__get__(model.bbox_head)
        model.bbox_head.get_bboxes = get_bboxes_ssd_head.__get__(model.bbox_head)
        model.onnx_export(img=data['img'][0],
                          img_meta=data['img_meta'][0],
                          export_name=export_name,
                          verbose=verbose,
                          opset_version=opset,
                          strip_doc_string=strip_doc_string,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          input_names=['image'],
                          output_names=['detection_out'])
    else:
        output_names = ['boxes', 'labels']
        dynamic_axes = {
            'image': {2: 'height', 3: 'width'},
            'boxes': {0: 'objects_num'},
            'labels': {0: 'objects_num'}
        }
        if model.with_mask:
            output_names.append('masks')
            dynamic_axes['masks'] = {0: 'objects_num'}

        with torch.no_grad():
            model.export(
                **data,
                export_name=export_name,
                verbose=verbose,
                opset_version=opset,
                strip_doc_string=strip_doc_string,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                input_names=['image'],
                output_names=output_names,
                dynamic_axes=dynamic_axes)


def check_onnx_model(export_name):
    model = onnx.load(export_name)
    try:
        onnx.checker.check_model(model)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed.')
        print(ex)


def add_node_names(export_name):
    model = onnx.load(export_name)
    for n in model.graph.node:
        if not n.name:
            n.name = '_'.join([i for i in n.output])
    onnx.save(model, export_name)


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = ','.join(out.name for out in onnx_model.graph.output)

    assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
    normalize = [v for v in cfg.data.test.pipeline[1]['transforms']
                 if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"' 
    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if normalize['to_rgb']:
        command_line += ' --reverse_input_channels'

    try:
        check_call('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True)
    except CalledProcessError as ex:
        print('OpenVINO Model Optimizer not found, please source '
              'openvino/bin/setupvars.sh before running this script.')
        return

    print(command_line)
    call(command_line, shell=True)


class AnchorsGridGeneratorStub(TracerStub):
    def __init__(self, inner, namespace='mmdet_custom',
                 name='anchor_grid_generator', **kwargs):
        super().__init__(inner, namespace=namespace, name=name, **kwargs)
        self.inner = lambda x, **kw: inner(x.shape[2:4], **kw)

    def forward(self, featmap, stride=16, device='cpu'):
        # Force `stride` and `device` to be passed in as kwargs.
        return super().forward(featmap, stride=stride, device=device)

    def symbolic(self, g, featmap):
        stride = float(self.params['stride'])
        shift = torch.full(self.params['base_anchors'].shape, - 0.5 * stride, dtype=torch.float32)
        prior_boxes = g.op('Constant', value_t=torch.tensor(self.params['base_anchors'], dtype=torch.float32) + shift)
        # TODO. im_data is not needed actually.
        im_data = g.op('Constant', value_t=torch.zeros([1, 1, 1, 1], dtype=torch.float32))

        return g.op('ExperimentalDetectronPriorGridGenerator',
                    prior_boxes,
                    featmap,
                    im_data,
                    stride_x_f=stride,
                    stride_y_f=stride,
                    h_i=0,
                    w_i=0,
                    flatten_i=1,
                    outputs=self.num_outputs)

def stub_anchor_generator(model, anchor_head_name):
    anchor_head = getattr(model, anchor_head_name, None)
    if anchor_head is not None and isinstance(anchor_head, AnchorHead):
        anchor_generators = anchor_head.anchor_generators
        for i in range(len(anchor_generators)):
            anchor_generators[i].grid_anchors = AnchorsGridGeneratorStub(
                anchor_generators[i].grid_anchors)
            # Save base anchors as operation parameter. It's used at ONNX export time during symbolic call.
            anchor_generators[i].grid_anchors.params['base_anchors'] = anchor_generators[
                i].base_anchors.cpu().numpy()


class ROIFeatureExtractorStub(TracerStub):

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, rois, *feats):
            return self.inner(feats, rois)

        def __getattr__(self, item):
            if item == 'inner':
                return super().__getattr__(item)
            # Forward calls to self.inner.
            inner = self.inner
            try:
                v = inner.__getattribute__(item)
            except AttributeError:
                v = inner.__getattr__(item)
            return v

    def __init__(self, inner, namespace='mmdet_custom',
                 name='roi_feature_extractor', **kwargs):
        super().__init__(self.Wrapper(inner), namespace=namespace, name=name, **kwargs)
        out_size = self.inner.roi_layers[0].out_size[0]
        for roi_layer in self.inner.roi_layers:
            size = roi_layer.out_size
            assert isinstance(roi_layer, RoIAlign)
            assert len(size) == 2
            assert size[0] == size[1]
            assert size[0] == out_size

    def forward(self, feats, rois, roi_scale_factor=None):
        assert roi_scale_factor is None, 'roi_scale_factor is not supported'
        return super().forward(rois, *feats)

    def symbolic(self, g, rois, *feats):
        rois = sym_help._slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
        roi_feats, _ = g.op('ExperimentalDetectronROIFeatureExtractor',
            rois,
            *feats,
            output_size_i=self.inner.roi_layers[0].out_size[0],
            pyramid_scales_i=self.inner.featmap_strides,
            sampling_ratio_i=self.inner.roi_layers[0].sample_num,
            image_id_i=0,
            distribute_rois_between_levels_i=1,
            preserve_rois_order_i=1,
            outputs=2
            )
        return roi_feats


def stub_roi_feature_extractor(model, extractor_name):
    if hasattr(model, extractor_name):
         extractor = getattr(model, extractor_name)
         if isinstance(extractor, SingleRoIExtractor):
             setattr(model, extractor_name, ROIFeatureExtractorStub(extractor))


def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data


def main(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    model = init_detector(args.config, args.checkpoint, device='cpu')
    model.eval()
    model.cuda()
    device = next(model.parameters()).device
    cfg = model.cfg
    fake_data = get_fake_input(cfg, device=device)

    if args.target == 'openvino' and not args.alt_ssd_export:
        stub_anchor_generator(model, 'rpn_head')
        stub_anchor_generator(model, 'bbox_head')
        stub_roi_feature_extractor(model, 'bbox_roi_extractor')
        stub_roi_feature_extractor(model, 'mask_roi_extractor')

    onnx_model_path = osp.join(args.output_dir,
                               osp.splitext(osp.basename(args.config))[0] + '.onnx')
    
    with torch.no_grad():
        export_to_onnx(model, fake_data, export_name=onnx_model_path, opset=10, alt_ssd_export=args.alt_ssd_export)
        add_node_names(onnx_model_path)
        print(f'ONNX model has been saved to "{onnx_model_path}"')

    if args.target == 'openvino':
        input_shape = list(fake_data['img'][0].shape)
        if args.input_shape:
            input_shape = [1, 3, *args.input_shape]
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape)
    else:
        check_onnx_model(onnx_model_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')

    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input_shape', nargs=2, type=int, default=None,
                                 help='input shape as a height-width pair')
    parser_openvino.add_argument('--alt_ssd_export', action='store_true',
                                 help='use alternative ONNX representation of SSD net')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
