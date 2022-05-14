# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from functools import partial

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.utils.det_cam_visualizer import (DetAblationLayer,
                                            DetBoxScoreTarget, DetCAMModel,
                                            DetCAMVisualizer, FeatmapAM,
                                            reshape_transform)

try:
    from pytorch_grad_cam import AblationCAM, EigenCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    'featmapam': FeatmapAM
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--method',
        default='featmapam',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-layers',
        default=['neck'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the neck')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Topk of the predicted result to visualizer')
    parser.add_argument(
        '--max-reshape-shape',
        type=tuple,
        default=(20, 20),
        help='max reshape shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to (-1, -1), it means no scaling.')
    parser.add_argument(
        '--norm-in-bbox',
        action='store_true',
        help='No norm in bbox of cam image')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument('--out-dir', default=None, help='dir to output file')

    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
        'The parameter controls how many channels should be ablated')

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
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def init_model_cam(args, cfg):
    # build the model from a config file and a checkpoint file
    model = DetCAMModel(
        cfg, args.checkpoint, args.score_thr, device=args.device)
    if args.preview_model:
        print(model.detector)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    det_cam_visualizer = DetCAMVisualizer(
        args.method,
        model,
        target_layers,
        batch_size=args.batch_size,
        reshape_transform=partial(
            reshape_transform, max_shape=args.max_reshape_shape),
        ablation_layer=DetAblationLayer(),
        ratio_channels_to_ablate=args.ratio_channels_to_ablate)
    return model, det_cam_visualizer


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model, det_cam_visualizer = init_model_cam(args, cfg)

    images = args.img
    if not isinstance(images, list):
        images = [images]

    for image_path in images:
        image = cv2.imread(image_path)
        model.set_input_data(image)
        result = model()[0]

        bboxes = result['bboxes'][..., :4]
        scores = result['bboxes'][..., 4]
        labels = result['labels']
        segms = result['segms']
        assert bboxes is not None and len(bboxes) > 0
        if args.topk > 0:
            idxs = np.argsort(-scores)
            bboxes = bboxes[idxs[:args.topk]]
            labels = labels[idxs[:args.topk]]
            if segms is not None:
                segms = segms[idxs[:args.topk]]
        targets = [
            DetBoxScoreTarget(bboxes=bboxes, labels=labels, segms=segms)
        ]
        grayscale_cam = det_cam_visualizer(
            image,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth)
        image_with_bounding_boxes = det_cam_visualizer.show_cam(
            image, bboxes, labels, grayscale_cam, args.norm_in_bbox)

        if args.out_dir:
            mmcv.mkdir_or_exist(args.out_dir)
            out_file = os.path.join(args.out_dir, os.path.basename(image_path))
            mmcv.imwrite(image_with_bounding_boxes, out_file)
        else:
            cv2.namedWindow(os.path.basename(image_path), 0)
            cv2.imshow(os.path.basename(image_path), image_with_bounding_boxes)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
