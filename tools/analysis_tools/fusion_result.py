# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
from mmengine.fileio import dump, load
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, check_file_exist

from mmdet.models.layers import weighted_boxes_fusion


def parse_args():
    parser = argparse.ArgumentParser(description='Fusion image prediction '
                                     'results from multiple models.')
    parser.add_argument(
        'prediction_path',
        type=str,
        nargs='+',
        help='prediction path where test pkl result.')
    parser.add_argument(
        '--weights',
        type=float,
        nargs='*',
        default=None,
        help='weights for each model, '
        'remember to correspond to the above prediction path.')
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
        default='avg',
        help='how to calculate confidence in weighted boxes in wbf')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    predicts = []
    fusion_predicts = []
    for path in args.prediction_path:
        check_file_exist(path)
        predicts.append(load(path))

    prog_bar = ProgressBar(len(predicts[0]))
    for pred in zip(*predicts):
        paths = np.array([pr['img_path'] for pr in pred])
        assert len(np.unique(paths)) == 1, 'prediction results are not match'

        bboxes_list = []
        scores_list = []
        labels_list = []

        for pr in pred:
            bboxes_list.append(pr['pred_instances']['bboxes'])
            scores_list.append(pr['pred_instances']['scores'])
            labels_list.append(pr['pred_instances']['labels'])

        bboxes, scores, labels = weighted_boxes_fusion(
            bboxes_list,
            scores_list,
            labels_list,
            weights=args.weights,
            iou_thr=args.fusion_iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type=args.conf_type)

        fusion_pred = dict(
            img_id=pred[0]['img_id'],
            ori_shape=pred[0]['ori_shape'],
            img_path=pred[0]['img_path'],
            pred_instances=dict(bboxes=bboxes, scores=scores, labels=labels))

        fusion_predicts.append(fusion_pred)
        prog_bar.update()

    out_file = args.out_dir + '/fusion_results.pkl'
    dump(fusion_predicts, file=out_file)

    print_log(
        f'Fusion results have been saved to {out_file}.', logger='current')


if __name__ == '__main__':
    main()
