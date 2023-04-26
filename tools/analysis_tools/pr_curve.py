# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mmengine.utils import mkdir_or_exist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOU_THR_INDEX = {
    0.5: 0,
    0.55: 1,
    0.6: 2,
    0.65: 3,
    0.7: 4,
    0.75: 5,
    0.8: 6,
    0.85: 7,
    0.9: 8,
    0.95: 9
}


def parse_args():
    parser = argparse.ArgumentParser(description='Parser PR Curve')
    parser.add_argument('ann_file', type=str, help='Annotation file path')
    parser.add_argument(
        'json_results',
        type=str,
        nargs='+',
        help='path of results in json (coco) format')
    parser.add_argument(
        '--iou_thrs',
        type=float,
        nargs='+',
        default=None,
        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, None],
        help='legend of each plot')
    parser.add_argument(
        '--title_base',
        type=str,
        default='',
        help='base title name adding before plot title')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot or title of each single result image')
    parser.add_argument(
        '--backend', default=None, type=str, help='backend of plt')
    parser.add_argument(
        '--style', default='dark', type=str, help='style of plt')
    parser.add_argument(
        '--classwise',
        action='store_true',
        help='Whether parse PR Curve class-wise')
    parser.add_argument(
        '--plot_single',
        action='store_true',
        help='Whether parse PR Curve in a result with different iou_thr')
    parser.add_argument(
        '--metric',
        default='bbox',
        type=str,
        choices=['bbox', 'segm'],
        help='metrics to be evaluated')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def get_precisions(coco_gt, json_results, labels, args):
    # defaults to [0.5:0.05:0.95]
    iou_thrs = [round(0.5 + 0.05 * x, 2) for x in range(10)]
    legend = []
    legend50 = []
    single_title = []
    if args.legend is None:
        for json_result in json_results:
            name = os.path.split(json_result)[-1].split('.json')[0]
            legend.append(f'{name}' + '(AP={})')
            legend50.append(f'{name}' + '(AP50={})')
            single_title.append(f'{name}')
    else:
        for _legend in args.legend:
            legend.append(f'{_legend}' + '(AP={})')
            single_title.append(f'{_legend}')
            legend50.append(f'{_legend}' + '(AP50={})')
    assert len(legend) == len(json_results)

    # prepare for the loop
    # `base_pr_array` is a base list used to quickly obtain`temp_mean_array`
    # in the loop
    base_pr_array = [[[] for _ in range(len(iou_thrs))]
                     for _ in range(len(labels))]
    # `base_mean_array` is a base list used to quickly obtain `temp_mean_array`
    # in the loop
    base_mean_array = [[] for _ in range(len(iou_thrs))]
    # in the loop
    base_ap = [[[] for _ in range(len(iou_thrs))] for _ in range(len(labels))]

    # `pr_arrays` is used to store processed single precision
    pr_arrays = []
    # `mean_arrays` is used to store processed mean precision of all labels
    mean_arrays = []
    # `base_ap` is a base list used to quickly obtain `temp_ap_list`

    # `ap_list` is used to store coco ap results
    ap_list = []
    # `ap50_list` is used to store coco ap50 results
    ap50_list = []
    # `label_ap_list` is used to store label ap results
    label_ap_list = []

    results_dict = dict()
    results_dict['legend'] = legend
    results_dict['legend50'] = legend50
    results_dict['single_title'] = single_title
    results_dict['iou_thrs'] = iou_thrs
    results_dict['json_results'] = json_results
    results_dict['labels'] = labels
    for i, json_result in enumerate(json_results):
        coco_dt = coco_gt.loadRes(json_result)
        coco_eval = COCOeval(coco_gt, coco_dt, args.metric)
        # `maxDets` defaults to 100, 300, 1000 in MMDetection,
        # while it defaults to 1, 10, 100 in Detectron2.
        # coco_eval.params.maxDets = list((1, 10, 100))
        coco_eval.params.maxDets = list((100, 300, 1000))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval['precision']
        # The meaning of precisions shape:
        # T: iou thresholds defaults to [0.5:0.05:0.95]
        # R: recall thresholds [0:0.1:1]
        # K: category
        # A: area range [all, small, medium, large]
        # M: max dets (0, 10, 100)
        temp_pr_array = copy.deepcopy(base_pr_array)
        temp_ap = copy.deepcopy(base_ap)
        temp_mean_array = copy.deepcopy(base_mean_array)
        for c in range(len(labels)):
            for iou_thr in iou_thrs:
                thr_id = IOU_THR_INDEX[iou_thr]
                temp_mean_array[thr_id].append(precisions[thr_id, :, c, 0, 2])
                temp_pr_array[c][thr_id] = precisions[thr_id, :, c, 0, 2]
                label_precision = copy.deepcopy(precisions[thr_id, :, c, 0, 2])
                label_precision = label_precision[label_precision > -1]
                if label_precision.size:
                    label_ap = np.mean(label_precision)
                else:
                    label_ap = float('nan')
                temp_ap[c][thr_id] = label_ap

        temp_mean = copy.deepcopy(base_mean_array)
        # calculate mean ap under different iou threshold
        for iou_thr in iou_thrs:
            thr_id = IOU_THR_INDEX[iou_thr]
            mean = np.vstack(temp_mean_array[thr_id]).mean(axis=0)
            temp_mean[thr_id] = mean

        pr_arrays.append(temp_pr_array)
        mean_arrays.append(temp_mean)
        label_ap_list.append(temp_ap)
        ap_list.append(coco_eval.stats[0])
        ap50_list.append(coco_eval.stats[1])
    results_dict['pr'] = pr_arrays
    results_dict['mpr'] = mean_arrays
    results_dict['ap'] = label_ap_list
    results_dict['map'] = ap_list
    results_dict['map50'] = ap50_list
    return results_dict


def plot_curve(results_dict, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # parse necessary args
    title_base = args.title_base
    if title_base is None:
        title_base = ''
    iou_thrs = args.iou_thrs
    if iou_thrs is None:
        iou_thrs = results_dict['iou_thrs']
    elif isinstance(iou_thrs, float):
        iou_thrs = [iou_thrs]
    else:
        assert isinstance(iou_thrs, list)

    # parse results_dict
    pr = results_dict['pr']
    mpr = results_dict['mpr']
    ap = results_dict['ap']
    map = results_dict['map']
    map50 = results_dict['map50']

    legend = results_dict['legend']
    legend50 = results_dict['legend50']
    single_title = results_dict['single_title']
    json_results = results_dict['json_results']
    labels = results_dict['labels']

    # x-axis coordinate: [0.00, 0.01, 0.02, ..., 1.00]
    x = np.arange(0.0, 1.01, 0.01)
    # plot curve in same results with different iou threshold
    if args.plot_single:
        for i, mpr_single in enumerate(mpr):
            for iou_thr in results_dict['iou_thrs']:
                label_name = f'IoU thr={iou_thr}'
                thr_id = IOU_THR_INDEX[iou_thr]
                plt.plot(
                    x, mpr_single[thr_id], label=label_name, linewidth=0.8)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc='lower left', fontsize='small')
            plt.title(f'{title_base} {single_title[i]} ')
            if args.out is None:
                plt.show()
            else:
                save_path = os.path.join(
                    args.out, f'single{title_base}_{single_title[i]}.png')
                print(f'Save single result curve to: {save_path}')
                plt.savefig(save_path)
            plt.cla()

    # plot curve in different results with same iou threshold
    if len(json_results) > 1:
        for iou_thr in iou_thrs:
            thr_id = IOU_THR_INDEX[iou_thr]
            for i, mpr_single in enumerate(mpr):
                if iou_thr == 0.5:
                    label_name = legend50[i].format(round(map50[i] * 100, 2))
                else:
                    label_name = legend[i].format(round(map[i] * 100, 2))
                plt.plot(
                    x, mpr_single[thr_id], label=label_name, linewidth=0.8)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc='lower left', fontsize='small')
            plt.title(f'{title_base} IoU Threshold = {iou_thr} ')
            if args.out is None:
                plt.show()
            else:
                save_path = os.path.join(
                    args.out, f'{title_base}iou_thr{int(iou_thr * 100)}.png')
                print(f'Save single result curve to: {save_path}')
                plt.savefig(save_path)
            plt.cla()

    if args.classwise:
        # plot curve in same results with different iou threshold
        if args.plot_single:
            for c, single_label in enumerate(labels):
                for i, pr_single in enumerate(pr):
                    for iou_thr in results_dict['iou_thrs']:
                        label_name = f'IoU thr={iou_thr}'
                        thr_id = IOU_THR_INDEX[iou_thr]
                        plt.plot(
                            x,
                            pr_single[c][thr_id],
                            label=label_name,
                            linewidth=0.8)
                    plt.xlabel('recall')
                    plt.ylabel('precision')
                    plt.xlim(0, 1.0)
                    plt.ylim(0, 1.01)
                    plt.grid(True)
                    plt.legend(loc='lower left', fontsize='small')
                    plt.title(f'{title_base} {single_title[i]} '
                              f'(category: {single_label})')
                    if args.out is None:
                        plt.show()
                    else:
                        save_path = os.path.join(
                            args.out, f'single{title_base}_{single_title[i]}'
                            f'_{single_label}.png')
                        print(f'Save single result curve to: {save_path}')
                        plt.savefig(save_path)
                    plt.cla()

        # plot curve in different results with same iou threshold
        if len(json_results) > 1:
            for c, single_label in enumerate(labels):
                for iou_thr in iou_thrs:
                    thr_id = IOU_THR_INDEX[iou_thr]
                    for i, pr_single in enumerate(pr):
                        if iou_thr == 0.5:
                            label_name = legend50[i].format(
                                round(ap[i][c][thr_id] * 100, 2))
                        else:
                            # get mean AP of the current category
                            label_name = legend[i].format(
                                round(np.array(ap[0][0]).mean() * 100, 2))
                        plt.plot(
                            x,
                            pr_single[c][thr_id],
                            label=label_name,
                            linewidth=0.8)
                    plt.xlabel('recall')
                    plt.ylabel('precision')
                    plt.xlim(0, 1.0)
                    plt.ylim(0, 1.01)
                    plt.grid(True)
                    plt.legend(loc='lower left', fontsize='small')
                    plt.title(f'{title_base} IoU Threshold = {iou_thr} '
                              f'(category: {single_label})')
                    if args.out is None:
                        plt.show()
                    else:
                        save_path = os.path.join(
                            args.out,
                            f'{title_base}iou_thr{int(iou_thr * 100)}_'
                            f'{single_label}.png')
                        print(f'Save single result curve to: {save_path}')
                        plt.savefig(save_path)
                    plt.cla()


def main():
    args = parse_args()

    json_results = args.json_results
    for json_result in json_results:
        assert json_result.endswith('.json'), \
            'Only support json format, please run the evaluate and ' \
            'set `CocoMetric.format_only=True` and ' \
            '`CocoMetric.outfile_prefix=xxx` ' \
            'to get json result first.'
    if args.plot_single:
        assert args.legend is not None and \
               len(json_results) == len(args.legend)

    if args.out is not None:
        mkdir_or_exist(args.out)

    coco_gt = COCO(annotation_file=args.ann_file)
    labels = [x['name'] for x in coco_gt.cats.values()]
    results_dict = get_precisions(coco_gt, json_results, labels, args)
    plot_curve(results_dict, args)


if __name__ == '__main__':
    main()
