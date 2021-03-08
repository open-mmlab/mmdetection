import argparse
import os.path as osp
import warnings

import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv import DictAction

from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)


def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None,
                 dataset='coco',
                 test_img=None,
                 do_simplify=False,
                 cfg_options=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the onnx model
    orig_model = build_model_from_cfg(
        config_path, checkpoint_path, cfg_options=cfg_options)
    one_img, one_meta = preprocess_example_input(input_config)
    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config, cfg_options=cfg_options)
    output_names = ['boxes']
    if model.with_bbox:
        output_names.append('labels')
    if model.with_mask:
        output_names.append('masks')

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=['input'],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version)

    model.forward = orig_model.forward

    # simplify onnx model
    if do_simplify:
        from mmdet import digit_version
        import mmcv

        min_required_version = '1.2.5'
        assert digit_version(mmcv.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install mmcv>={min_required_version}'
        from mmcv.onnx.simplify import simplify

        input_dic = {'input': one_img.detach().cpu().numpy()}
        _ = simplify(output_file, [input_dic], output_file)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        from mmdet.core import get_classes, bbox2result
        from mmdet.apis import show_result_pyplot

        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with ONNXRuntime from source.')
        model.CLASSES = get_classes(dataset)
        num_classes = len(model.CLASSES)
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        if test_img is not None:
            input_config['input_path'] = test_img
            one_img, one_meta = preprocess_example_input(input_config)
            tensor_data = [one_img]
        # check the numerical value
        # get pytorch output
        pytorch_results = model(tensor_data, [[one_meta]], return_loss=False)
        pytorch_results = pytorch_results[0]
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        session_options = rt.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = rt.InferenceSession(output_file, session_options)
        onnx_outputs = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})
        output_names = [_.name for _ in sess.get_outputs()]
        output_shapes = [_.shape for _ in onnx_outputs]
        print(f'onnxruntime output names: {output_names}, \
            output shapes: {output_shapes}')
        nrof_out = len(onnx_outputs)
        assert nrof_out > 0, 'Must have output'
        with_mask = nrof_out == 3
        if nrof_out == 1:
            onnx_results = onnx_outputs[0]
        else:
            det_bboxes, det_labels = onnx_outputs[:2]
            onnx_results = bbox2result(det_bboxes, det_labels, num_classes)
            if with_mask:
                segm_results = onnx_outputs[2].squeeze(1)
                cls_segms = [[] for _ in range(num_classes)]
                for i in range(det_bboxes.shape[0]):
                    cls_segms[det_labels[i]].append(segm_results[i])
                onnx_results = (onnx_results, cls_segms)
        # visualize predictions

        if show:
            show_result_pyplot(
                model, one_meta['show_img'], pytorch_results, title='Pytorch')
            show_result_pyplot(
                model, one_meta['show_img'], onnx_results, title='ONNX')

        # compare a part of result

        if with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res,
                    p_res,
                    rtol=1e-03,
                    atol=1e-05,
                )
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
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
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
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../../tests/data/color.jpg')

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
        dataset=args.dataset,
        test_img=args.test_img,
        do_simplify=args.simplify,
        cfg_options=args.cfg_options)
