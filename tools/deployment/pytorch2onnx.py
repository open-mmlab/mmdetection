import argparse
import copy
import os.path as osp
import warnings
from functools import partial

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def _prepare_inputs(img_path, pipeline, shape=None, update_params=True):
    test_pipeline = copy.deepcopy(pipeline)
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]['img_scale'] = (shape[1], shape[0])

    if update_params:
        # update parameters in transforms
        for trans in test_pipeline[1]['transforms']:
            if 'keep_ratio' in trans:
                trans['keep_ratio'] = False
            if 'size_divisor' in trans:
                trans['size_divisor'] = 1
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    imgs = data['img']
    img_metas = [[i.data] for i in data['img_metas']]
    imgs = [img[None, :] for img in imgs]
    if update_params:
        for img_meta in img_metas:
            img_meta[0]['scale_factor'] = 1.0
            img_meta[0]['flip'] = False
    return imgs, img_metas


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 test_pipeline,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None):

    model.cpu().eval()
    img_list, img_meta_list = _prepare_inputs(
        input_img, test_pipeline, shape=input_shape)

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    output_names = ['dets', 'labels']
    if model.with_mask:
        output_names.append('masks')
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    torch.onnx.export(
        model,
        img_list,
        output_file,
        input_names=['input'],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    model.forward = origin_forward

    # get the custom op path
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')

    if do_simplify:
        from mmdet import digit_version
        import onnxsim

        min_required_version = '0.3.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        input_dic = {'input': img_list[0].detach().cpu().numpy()}
        onnxsim.simplify(
            output_file, input_data=input_dic, custom_lib=ort_custom_op_path)
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        from mmdet.core import bbox2result
        from mmdet.apis import show_result_pyplot
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        if dynamic_export:
            # scale up to test dynamic shape
            input_shape = [int((_ * 1.5) // 32 * 32) for _ in input_shape]

        if test_img is None:
            test_img = input_img

        # prepare test inputs
        img_list, img_meta_list = _prepare_inputs(
            test_img, test_pipeline, shape=input_shape)

        # get pytorch output
        pytorch_results = model(
            img_list, img_metas=img_meta_list, return_loss=False)[0]
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        session_options = rt.SessionOptions()
        # register custom op for ONNX Runtime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        feed_input_img = img_list[0].detach().numpy()
        if dynamic_export:
            # test batch with two input images
            feed_input_img = np.vstack([feed_input_img, feed_input_img])
        sess = rt.InferenceSession(output_file, session_options)
        onnx_outputs = sess.run(None, {net_feed_input[0]: feed_input_img})
        output_names = [_.name for _ in sess.get_outputs()]
        output_shapes = [_.shape for _ in onnx_outputs]
        print(f'ONNX Runtime output names: {output_names}, \
            output shapes: {output_shapes}')
        # get last image's outputs
        onnx_outputs = [_[-1] for _ in onnx_outputs]
        ort_dets, ort_labels = onnx_outputs[:2]
        num_classes = len(model.CLASSES)
        onnx_results = bbox2result(ort_dets, ort_labels, num_classes)
        if model.with_mask:
            segm_results = onnx_outputs[2]
            cls_segms = [[] for _ in range(num_classes)]
            for i in range(ort_dets.shape[0]):
                cls_segms[ort_labels[i]].append(segm_results[i])
            onnx_results = (onnx_results, cls_segms)
        # visualize predictions
        if show:
            show_img = mmcv.imread(test_img)
            h, w = img_meta_list[0][0]['img_shape'][:2]
            show_img = mmcv.imresize(show_img, (w, h))
            show_result_pyplot(
                model, show_img, pytorch_results, title='Pytorch')
            show_result_pyplot(
                model, show_img, onnx_results, title='ONNXRuntime')

        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
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
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = args.shape
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmdet.datasets import *  # noqa: F401, F403
        dataset = eval(cfg.data.test['type'])
        model.CLASSES = dataset.CLASSES

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        cfg.test_pipeline,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export)
