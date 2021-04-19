import argparse
import os.path as osp
import warnings

import mmcv
import numpy as np
import onnxruntime as ort
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test
from mmdet.core import bbox2result
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import BaseDetector


class ONNXRuntimeDetector(BaseDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, onnx_file, class_names, device_id):
        super(ONNXRuntimeDetector, self).__init__()
        # get the custom op path
        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        option = {'device_id': device_id}
        sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'],
                           [option, {}])

        self.sess = sess
        self.CLASSES = class_names
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, imgs, img_metas, **kwargs):
        input_data = imgs[0]
        img_metas = img_metas[0]
        batch_size = input_data.shape[0]
        # set io binding for inputs/outputs
        self.io_binding.bind_input(
            name='input',
            device_type='cuda',
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())
        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        ort_outputs = self.io_binding.copy_outputs_to_cpu()
        batch_dets, batch_labels = ort_outputs[:2]
        batch_masks = ort_outputs[2] if len(ort_outputs) == 3 else None

        results = []
        for i in range(batch_size):
            scale_factor = img_metas[i]['scale_factor']
            dets, labels = batch_dets[i], batch_labels[i]
            dets[:, :4] /= scale_factor
            dets_results = bbox2result(dets, labels, len(self.CLASSES))
            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                masks = masks[:, :img_h, :img_w]
                mask_dtype = masks.dtype
                masks = masks.astype(np.float32)
                masks = torch.from_numpy(masks)
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0), size=(ori_h, ori_w))
                masks = masks.squeeze(0).detach().numpy()
                # convert mask to range(0,1)
                if mask_dtype != np.bool:
                    masks /= 255
                masks = masks >= 0.5
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) an ONNX model using ONNXRuntime')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='Input model file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
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
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = ONNXRuntimeDetector(
        args.model, class_names=dataset.CLASSES, device_id=0)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                              args.show_score_thr)

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
