# Copyright (c) OpenMMLab. All rights reserved.
"""Support for multi-model fusion, and currently only the Weighted Box Fusion
(WBF) fusion method is supported.

References: https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Example:

     python demo/demo_multi_model.py demo/demo.jpg \
         ./configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py \
         ./configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py \
         --checkpoints \
         https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth \  # noqa
         https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth \
         --weights 1 2
"""

import argparse
import os.path as osp

import mmcv
import mmengine
from mmengine.fileio import isdir, join_path, list_dir_or_file
from mmengine.logging import print_log
from mmengine.structures import InstanceData

from mmdet.apis import DetInferencer
from mmdet.models.utils import weighted_boxes_fusion
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDetection multi-model inference demo')
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'config',
        type=str,
        nargs='*',
        help='Config file(s), support receive multiple files')
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='*',
        help='Checkpoint file(s), support receive multiple files, '
        'remember to correspond to the above config',
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs='*',
        default=None,
        help='weights for each model, remember to '
        'correspond to the above config')
    parser.add_argument(
        '--fusion-iou-thr',
        type=float,
        default=0.55,
        help='IoU value for boxes to be a match in wbf')
    parser.add_argument(
        '--skip-box-thr',
        type=float,
        default=0.0,
        help='exclude boxes with score lower than this variable in wbf')
    parser.add_argument(
        '--conf-type',
        type=str,
        default='avg',  # avg, max, box_and_model_avg, absent_model_aware_avg
        help='how to calculate confidence in weighted boxes in wbf')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')

    args = parser.parse_args()

    if args.no_save_vis and args.no_save_pred:
        args.out_dir = ''

    return args


def main():
    args = parse_args()

    results = []
    cfg_visualizer = None
    dataset_meta = None

    inputs = []
    filename_list = []
    if isdir(args.inputs):
        dir = list_dir_or_file(
            args.inputs, list_dir=False, suffix=IMG_EXTENSIONS)
        for filename in dir:
            img = mmcv.imread(join_path(args.inputs, filename))
            inputs.append(img)
            filename_list.append(filename)
    else:
        img = mmcv.imread(args.inputs)
        inputs.append(img)
        img_name = osp.basename(args.inputs)
        filename_list.append(img_name)

    for i, (config,
            checkpoint) in enumerate(zip(args.config, args.checkpoints)):
        inferencer = DetInferencer(
            config, checkpoint, device=args.device, palette=args.palette)

        result_raw = inferencer(
            inputs=inputs,
            batch_size=args.batch_size,
            no_save_vis=True,
            pred_score_thr=args.pred_score_thr)

        if i == 0:
            cfg_visualizer = inferencer.cfg.visualizer
            dataset_meta = inferencer.model.dataset_meta
            results = [{
                'bboxes_list': [],
                'scores_list': [],
                'labels_list': []
            } for _ in range(len(result_raw['predictions']))]

        for res, raw in zip(results, result_raw['predictions']):
            res['bboxes_list'].append(raw['bboxes'])
            res['scores_list'].append(raw['scores'])
            res['labels_list'].append(raw['labels'])

    visualizer = VISUALIZERS.build(cfg_visualizer)
    visualizer.dataset_meta = dataset_meta

    for i in range(len(results)):
        bboxes, scores, labels = weighted_boxes_fusion(
            results[i]['bboxes_list'],
            results[i]['scores_list'],
            results[i]['labels_list'],
            weights=args.weights,
            iou_thr=args.fusion_iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type=args.conf_type)

        pred_instances = InstanceData()
        pred_instances.bboxes = bboxes
        pred_instances.scores = scores
        pred_instances.labels = labels

        fusion_result = DetDataSample(pred_instances=pred_instances)

        img_name = filename_list[i]

        if not args.no_save_pred:
            out_json_path = (
                args.out_dir + '/preds/' + img_name.split('.')[0] + '.json')
            mmengine.dump(
                {
                    'labels': labels.tolist(),
                    'scores': scores.tolist(),
                    'bboxes': bboxes.tolist()
                }, out_json_path)

        out_file = osp.join(args.out_dir, 'vis',
                            img_name) if not args.no_save_vis else None

        visualizer.add_datasample(
            img_name,
            inputs[i][..., ::-1],
            data_sample=fusion_result,
            show=args.show,
            draw_gt=False,
            wait_time=0,
            pred_score_thr=args.pred_score_thr,
            out_file=out_file)

    if not args.no_save_vis:
        print_log(f'results have been saved at {args.out_dir}')


if __name__ == '__main__':
    main()
