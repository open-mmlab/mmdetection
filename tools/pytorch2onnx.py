import argparse
import os
import os.path as osp

import numpy as np
import onnx
import onnxruntime as rt
import torch
from onnx import optimizer
from onnxsim import simplify

from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)


def optimize_model(model_path, output_path=None, passes=None):
    passes = [] if passes is None else passes
    output_path = model_path if output_path is None else output_path
    out_dir, _ = os.path.split(output_path)
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(model_path)
    opt_model = optimizer.optimize(model, passes)
    onnx.save(opt_model, output_path)


def slim_model(model_path, output_path=None):
    output_path = model_path if output_path is None else output_path
    out_dir, _ = os.path.split(output_path)
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(model_path)
    opt_model, check = simplify(model)
    assert check, f'Simplified ONNX model could not be validated: {model_path}'
    onnx.save(opt_model, output_path)


def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None,
                 dataset='coco'):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the onnx model
    orig_model = build_model_from_cfg(config_path, checkpoint_path)
    one_img, one_meta = preprocess_example_input(input_config)

    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version)

    model.forward = orig_model.forward
    optimize_model(output_file)
    slim_model(output_file)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        from mmdet.core import get_classes
        from mmdet.apis import show_result_pyplot
        model.CLASSES = get_classes(dataset)
        num_classes = len(model.CLASSES)
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(tensor_data, [[one_meta]], return_loss=False)
        pytorch_results = pytorch_results[0]
        assert np.asarray(
            pytorch_results
        ).size > 0, 'No pytorch results, consider change input image.'

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        from mmdet.core import bbox2result
        onnx_outputs = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})
        det_bboxes, det_labels = onnx_outputs[:2]
        onnx_results = bbox2result(det_bboxes, det_labels, num_classes)
        with_mask = len(onnx_outputs) == 3
        if with_mask:
            pass
            # TODO add mask pose processing

        # visualize predictions
        if args.view:
            show_result_pyplot(
                model,
                one_meta['show_img'],
                pytorch_results,
                title='Pytorch',
                block=False)
            show_result_pyplot(
                model, one_meta['show_img'], onnx_results, title='ONNX')

        # compare a part of result
        if with_mask:
            compare_pairs = list(zip(onnx_results[0], pytorch_results[0]))
        else:
            compare_pairs = [(onnx_results[0], pytorch_results[0])]

        for onnx_res, pytorch_res in compare_pairs:
            np.testing.assert_allclose(
                onnx_res,
                pytorch_res,
                rtol=1e-05,
                err_msg='The outputs are different between Pytorch and ONNX')
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--view', action='store_true', help='Visualize results')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset)
