import os.path as osp
from argparse import ArgumentParser

import mmcv
import numpy as np


def print_coco_results(results):

    def _print(result, ap=1, iouThr=None, areaRng='all', maxDets=100):
        iStr = ' {:<18} {} @[ IoU={:<9} | \
        area={:>6s} | maxDets={:>3d} ] = {:0.3f}'

        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(.5, .95) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, result))

    stats = np.zeros((12, ))
    stats[0] = _print(results[0], 1)
    stats[1] = _print(results[1], 1, iouThr=.5)
    stats[2] = _print(results[2], 1, iouThr=.75)
    stats[3] = _print(results[3], 1, areaRng='small')
    stats[4] = _print(results[4], 1, areaRng='medium')
    stats[5] = _print(results[5], 1, areaRng='large')
    stats[6] = _print(results[6], 0, maxDets=1)
    stats[7] = _print(results[7], 0, maxDets=10)
    stats[8] = _print(results[8], 0)
    stats[9] = _print(results[9], 0, areaRng='small')
    stats[10] = _print(results[10], 0, areaRng='medium')
    stats[11] = _print(results[11], 0, areaRng='large')


def get_coco_style_results(filename,
                           task='bbox',
                           metric=None,
                           prints='mPC',
                           aggregate='benchmark'):

    assert aggregate in ['benchmark', 'all']

    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']

    if metric is None:
        metrics = [
            'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100',
            'ARs', 'ARm', 'ARl'
        ]
    elif isinstance(metric, list):
        metrics = metric
    else:
        metrics = [metric]

    for metric_name in metrics:
        assert metric_name in [
            'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100',
            'ARs', 'ARm', 'ARl'
        ]

    eval_output = mmcv.load(filename)

    num_distortions = len(list(eval_output.keys()))
    results = np.zeros((num_distortions, 6, len(metrics)), dtype='float32')

    for corr_i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            for metric_j, metric_name in enumerate(metrics):
                mAP = eval_output[distortion][severity][task][metric_name]
                results[corr_i, severity, metric_j] = mAP

    P = results[0, 0, :]
    if aggregate == 'benchmark':
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P

    print('\nmodel: {}'.format(osp.basename(filename)))
    if metric is None:
        if 'P' in prints:
            print('Performance on Clean Data [P] ({})'.format(task))
            print_coco_results(P)
        if 'mPC' in prints:
            print('Mean Performance under Corruption [mPC] ({})'.format(task))
            print_coco_results(mPC)
        if 'rPC' in prints:
            print('Realtive Performance under Corruption [rPC] ({})'.format(
                task))
            print_coco_results(rPC)
    else:
        if 'P' in prints:
            print('Performance on Clean Data [P] ({})'.format(task))
            for metric_i, metric_name in enumerate(metrics):
                print('{:5} =  {:0.3f}'.format(metric_name, P[metric_i]))
        if 'mPC' in prints:
            print('Mean Performance under Corruption [mPC] ({})'.format(task))
            for metric_i, metric_name in enumerate(metrics):
                print('{:5} =  {:0.3f}'.format(metric_name, mPC[metric_i]))
        if 'rPC' in prints:
            print('Relative Performance under Corruption [rPC] ({})'.format(
                task))
            for metric_i, metric_name in enumerate(metrics):
                print('{:5} => {:0.1f} %'.format(metric_name,
                                                 rPC[metric_i] * 100))

    return results


def get_voc_style_results(filename, prints='mPC', aggregate='benchmark'):

    assert aggregate in ['benchmark', 'all']

    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']

    eval_output = mmcv.load(filename)

    num_distortions = len(list(eval_output.keys()))
    results = np.zeros((num_distortions, 6, 20), dtype='float32')

    for i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            mAP = [
                eval_output[distortion][severity][j]['ap']
                for j in range(len(eval_output[distortion][severity]))
            ]
            results[i, severity, :] = mAP

    P = results[0, 0, :]
    if aggregate == 'benchmark':
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P

    print('\nmodel: {}'.format(osp.basename(filename)))
    if 'P' in prints:
        print('{:48} = {:0.3f}'.format('Performance on Clean Data [P] in AP50',
                                       np.mean(P)))
    if 'mPC' in prints:
        print('{:48} = {:0.3f}'.format(
            'Mean Performance under Corruption [mPC] in AP50', np.mean(mPC)))
    if 'rPC' in prints:
        print('{:48} = {:0.1f}'.format(
            'Realtive Performance under Corruption [rPC] in %',
            np.mean(rPC) * 100))

    return np.mean(results, axis=2, keepdims=True)


def get_results(filename,
                dataset='coco',
                task='bbox',
                metric=None,
                prints='mPC',
                aggregate='benchmark'):
    assert dataset in ['coco', 'voc', 'cityscapes']

    if dataset in ['coco', 'cityscapes']:
        results = get_coco_style_results(
            filename,
            task=task,
            metric=metric,
            prints=prints,
            aggregate=aggregate)
    elif dataset == 'voc':
        if task != 'bbox':
            print('Only bbox analysis is supported for Pascal VOC')
            print('Will report bbox results\n')
        if metric not in [None, ['AP'], ['AP50']]:
            print('Only the AP50 metric is supported for Pascal VOC')
            print('Will report AP50 metric\n')
        results = get_voc_style_results(
            filename, prints=prints, aggregate=aggregate)

    return results


def get_distortions_from_file(filename):

    eval_output = mmcv.load(filename)

    return get_distortions_from_results(eval_output)


def get_distortions_from_results(eval_output):
    distortions = []
    for i, distortion in enumerate(eval_output):
        distortions.append(distortion.replace("_", " "))
    return distortions


def main():
    parser = ArgumentParser(description='Corruption Result Analysis')
    parser.add_argument('filename', help='result file path')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'voc', 'cityscapes'],
        default='coco',
        help='dataset type')
    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        default=['bbox'],
        help='task to report')
    parser.add_argument(
        '--metric',
        nargs='+',
        choices=[
            None, 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
            'AR100', 'ARs', 'ARm', 'ARl'
        ],
        default=None,
        help='metric to report')
    parser.add_argument(
        '--prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default='mPC',
        help='corruption benchmark metric to print')
    parser.add_argument(
        '--aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='benchmark',
        help='aggregate all results or only those \
        for benchmark corruptions')

    args = parser.parse_args()

    for task in args.task:
        get_results(
            args.filename,
            dataset=args.dataset,
            task=task,
            metric=args.metric,
            prints=args.prints,
            aggregate=args.aggregate)


if __name__ == '__main__':
    main()
