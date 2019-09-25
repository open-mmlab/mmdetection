import argparse

import numpy as np
import onnx
import torch
import torch.onnx.symbolic_helper as sym_help
from mmcv.parallel import collate, scatter
from torch.onnx.symbolic_helper import _unimplemented, parse_args
from torch.onnx.symbolic_registry import register_op

from mmdet.apis import init_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose


@parse_args('v', 'v', 'v', 'v', 'none')
def addcmul_symbolic(g, self, tensor1, tensor2, value=1, out=None):
    from torch.onnx.symbolic_opset9 import add, mul

    if out is not None:
        _unimplemented("addcmul", "Out parameter is not supported for addcmul")

    x = mul(g, tensor1, tensor2)
    value = sym_help._maybe_get_scalar(value)
    if sym_help._scalar(value) != 1:
        value = sym_help._if_scalar_type_as(g, value, x)
        if not sym_help._is_value(value):
            value = g.op(
                "Constant", value_t=torch.tensor(value, dtype=torch.float32))
        x = mul(g, x, value)
    return add(g, self, x)


def view_as_symbolic(g, self, other):
    from torch.onnx.symbolic_opset9 import reshape_as
    return reshape_as(g, self, other)


@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk_symbolic(g, self, k, dim, largest, sorted, out=None):

    def reverse(x):
        from torch.onnx.symbolic_opset9 import reshape, transpose, size

        y = transpose(g, x, 0, dim)
        shape = g.op("Shape", y)
        y = reshape(g, y, [0, 1, -1])
        n = size(g, y, g.op("Constant", value_t=torch.LongTensor([0])))
        y = g.op("ReverseSequence", y, n, batch_axis_i=1, time_axis_i=0)
        y = reshape(g, y, shape)
        y = transpose(g, y, 0, dim)
        return y

    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    k = sym_help._maybe_get_const(k, 'i')
    if not sym_help._is_value(k):
        k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
    from torch.onnx.symbolic_opset9 import unsqueeze
    k = unsqueeze(g, k, 0)
    top_values, top_indices = g.op("TopK", self, k, axis_i=dim, outputs=2)
    if not largest:
        top_values = reverse(top_values)
        top_indices = reverse(top_indices)
    return top_values, top_indices


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm_symbolic(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    return g.op("GroupNorm", input, weight, bias, num_groups_i=num_groups, eps_f=eps)


def export(model,
           img,
           export_name="/tmp/model.onnx",
           verbose=False,
           strip_doc_string=False):
    register_op("addcmul", addcmul_symbolic, "", 10)
    register_op("view_as", view_as_symbolic, "", 10)
    register_op("topk", topk_symbolic, "", 10)
    register_op("group_norm", group_norm_symbolic, "", 10)

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
    output_names = ["boxes", "labels"]
    dynamic_axes = {
        "image": {
            2: "height",
            3: "width"
        },
        "boxes": {
            0: "objects_num"
        },
        "labels": {
            0: "objects_num"
        }
    }
    if model.with_mask:
        output_names.append("masks")
        dynamic_axes["masks"] = {0: "objects_num"}
    with torch.no_grad():
        model.export(
            **data,
            export_name=export_name,
            verbose=verbose,
            opset_version=10,
            strip_doc_string=strip_doc_string,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=dynamic_axes)


def check_onnx_model(export_name):
    model = onnx.load(export_name)

    try:
        onnx.checker.check_model(model)
        print("ONNX check passed.")
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print("ONNX check failed.")
        print(ex)


def add_node_names(export_name):
    model = onnx.load(export_name)
    for n in model.graph.node:
        n.name = "in: " + ",".join([i for i in n.input]) + ". " + \
                 "out: " + ",".join([i for i in n.output])
    onnx.save(model, export_name)


from mmdet.utils.tracer_stubs import TracerStub


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
        prior_boxes = g.op("Constant", value_t=torch.tensor(self.params['base_anchors'], dtype=torch.float32) + shift)
        # TODO. im_data is not needed actually.
        im_data = g.op("Constant", value_t=torch.zeros([1, 1, 1, 1], dtype=torch.float32))

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


def main(args):
    model = init_detector(args.config, args.checkpoint, device="cpu")
    model.eval().cuda()
    torch.set_default_tensor_type(torch.FloatTensor)

    # class StubModule(TracerStub):
    #     def __init__(self, inner):
    #         super().__init__(inner, 'mmdet_custom', 'x')
    #
    #     def forward(self, *input, **kwargs):
    #         return self.inner(*input, **kwargs)[0]
    #
    # class Stub(TracerStub):
    #
    #     def symbolic(self, g, *args):
    #         return g.op(self.name, *args, outputs=self.num_outputs)

    # model.backbone = Stub(model.backbone, name='EfficientNet', namespace='mmdet_custom')
    # model.bbox_head = Stub(StubModule(model.bbox_head), name='DetectionOutput', namespace='mmdet_custom')
    # model.bbox_head.get_bboxes = Stub(model.bbox_head.get_bboxes, name='DetectionOutput', namespace='mmdet_custom')

    anchor_generators = model.bbox_head.anchor_generators
    for i in range(len(anchor_generators)):
        anchor_generators[i].grid_anchors = AnchorsGridGeneratorStub(anchor_generators[i].grid_anchors)
        # Save base anchors as operation parameter. It's used at ONNX export time during symbolic call.
        anchor_generators[i].grid_anchors.params['base_anchors'] = anchor_generators[i].base_anchors.cpu().numpy()

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with torch.no_grad():
        export(model, image, export_name=args.model)

    # add_node_names(args.model)

    if args.check:
        check_onnx_model(args.model)


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("model", help="path to output onnx model file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="check that resulting onnx model is valid")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="path to file with model's weights")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
