import argparse
import sys
from copy import copy

import numpy as np
import onnx
import torch
from mmcv.parallel import collate, scatter

from mmdet.apis import init_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmdet.models.anchor_heads.anchor_head import AnchorHead
from mmdet.utils.deployment import register_extra_symbolics, TracerStub


def export(model,
           img,
           export_name,
           verbose=False,
           strip_doc_string=False,
           opset=10):
    register_extra_symbolics(opset)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    output_names = ['boxes', 'labels']
    dynamic_axes = {
        'image': {
            2: 'height',
            3: 'width'
        },
        'boxes': {
            0: 'objects_num'
        },
        'labels': {
            0: 'objects_num'
        }
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
        n.name = 'in: ' + ','.join([i for i in n.input]) + '. ' + \
                 'out: ' + ','.join([i for i in n.output])
    onnx.save(model, export_name)


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

def stub_anchor_generator(anchor_head):
    if anchor_head is not None and isinstance(anchor_head, AnchorHead):
        anchor_generators = anchor_head.anchor_generators
        for i in range(len(anchor_generators)):
            anchor_generators[i].grid_anchors = AnchorsGridGeneratorStub(
                anchor_generators[i].grid_anchors)
            # Save base anchors as operation parameter. It's used at ONNX export time during symbolic call.
            anchor_generators[i].grid_anchors.params['base_anchors'] = anchor_generators[
                i].base_anchors.cpu().numpy()

def main(args):
    model = init_detector(args.config, args.checkpoint, device='cpu')
    model.eval().cuda()
    torch.set_default_tensor_type(torch.FloatTensor)

    if not args.no_stubs:
        stub_anchor_generator(getattr(model, 'rpn_head', None))
        stub_anchor_generator(getattr(model, 'bbox_head', None))
    
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with torch.no_grad():
        export(model, image, export_name=args.model, opset=10)

    # add_node_names(args.model)

    if args.check:
        check_onnx_model(args.model)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('model', help='path to output onnx model file')
    parser.add_argument('--check', action='store_true',
                        help='check that resulting onnx model is valid')
    parser.add_argument('--no_stubs', action='store_true',
                        help='disable all exporting stubs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
