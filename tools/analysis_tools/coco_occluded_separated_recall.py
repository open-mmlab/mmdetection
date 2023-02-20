# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmengine
from mmengine.logging import print_log

from mmdet.datasets import CocoDataset
from mmdet.evaluation import CocoOccludedSeparatedMetric


def main():
    parser = ArgumentParser(
        description='Compute recall of COCO occluded and separated masks '
        'presented in paper https://arxiv.org/abs/2210.10046.')
    parser.add_argument('result', help='result file (pkl format) path')
    parser.add_argument('--out', help='file path to save evaluation results')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold for the recall calculation. Defaults to 0.3')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.75,
        help='IoU threshold for the recall calculation. Defaults to 0.75.')
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/instances_val2017.json',
        help='coco annotation file path')
    args = parser.parse_args()

    results = mmengine.load(args.result)
    assert 'masks' in results[0]['pred_instances'], \
        'The results must be predicted by instance segmentation model.'
    metric = CocoOccludedSeparatedMetric(
        ann_file=args.ann, iou_thr=args.iou_thr, score_thr=args.score_thr)
    metric.dataset_meta = CocoDataset.METAINFO
    for datasample in results:
        metric.process(data_batch=None, data_samples=[datasample])
    metric_res = metric.compute_metrics(metric.results)
    if args.out is not None:
        mmengine.dump(metric_res, args.out)
        print_log(f'Evaluation results have been saved to {args.out}.')


if __name__ == '__main__':
    main()
