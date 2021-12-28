# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def makeplot(rs, ps, outDir, class_name, iou_type):
    cs = np.vstack([
        np.ones((2, 3)),
        np.array([0.31, 0.51, 0.74]),
        np.array([0.75, 0.31, 0.30]),
        np.array([0.36, 0.90, 0.38]),
        np.array([0.50, 0.39, 0.64]),
        np.array([1, 0.6, 0]),
    ])
    areaNames = ['allarea', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_title = iou_type + '-' + class_name + '-' + areaNames[i]
        aps = [ps_.mean() for ps_ in area_ps]
        ps_curve = [
            ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in area_ps
        ]
        ps_curve.insert(0, np.zeros(ps_curve[0].shape))
        fig = plt.figure()
        ax = plt.subplot(111)
        for k in range(len(types)):
            ax.plot(rs, ps_curve[k + 1], color=[0, 0, 0], linewidth=0.5)
            ax.fill_between(
                rs,
                ps_curve[k],
                ps_curve[k + 1],
                color=cs[k],
                label=str(f'[{aps[k]:.3f}]' + types[k]),
            )
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.title(figure_title)
        plt.legend()
        # plt.show()
        fig.savefig(outDir + f'/{figure_title}.png')
        plt.close(fig)


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0 and height <= 1:  # for percent values
            text_label = '{:2.0f}'.format(height * 100)
        else:
            text_label = '{:2.0f}'.format(height)
        ax.annotate(
            text_label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize='x-small',
        )


def makebarplot(rs, ps, outDir, class_name, iou_type):
    areaNames = ['allarea', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    fig, ax = plt.subplots()
    x = np.arange(len(areaNames))  # the areaNames locations
    width = 0.60  # the width of the bars
    rects_list = []
    figure_title = iou_type + '-' + class_name + '-' + 'ap bar plot'
    for i in range(len(types) - 1):
        type_ps = ps[i, ..., 0]
        aps = [ps_.mean() for ps_ in type_ps.T]
        rects_list.append(
            ax.bar(
                x - width / 2 + (i + 1) * width / len(types),
                aps,
                width / len(types),
                label=types[i],
            ))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Average Precision (mAP)')
    ax.set_title(figure_title)
    ax.set_xticks(x)
    ax.set_xticklabels(areaNames)
    ax.legend()

    # Add score texts over bars
    for rects in rects_list:
        autolabel(ax, rects)

    # Save plot
    fig.savefig(outDir + f'/{figure_title}.png')
    plt.close(fig)


def get_gt_area_group_numbers(cocoEval):
    areaRng = cocoEval.params.areaRng
    areaRngStr = [str(aRng) for aRng in areaRng]
    areaRngLbl = cocoEval.params.areaRngLbl
    areaRngStr2areaRngLbl = dict(zip(areaRngStr, areaRngLbl))
    areaRngLbl2Number = dict.fromkeys(areaRngLbl, 0)
    for evalImg in cocoEval.evalImgs:
        if evalImg:
            for gtIgnore in evalImg['gtIgnore']:
                if not gtIgnore:
                    aRngLbl = areaRngStr2areaRngLbl[str(evalImg['aRng'])]
                    areaRngLbl2Number[aRngLbl] += 1
    return areaRngLbl2Number


def make_gt_area_group_numbers_plot(cocoEval, outDir, verbose=True):
    areaRngLbl2Number = get_gt_area_group_numbers(cocoEval)
    areaRngLbl = areaRngLbl2Number.keys()
    if verbose:
        print('number of annotations per area group:', areaRngLbl2Number)

    # Init figure
    fig, ax = plt.subplots()
    x = np.arange(len(areaRngLbl))  # the areaNames locations
    width = 0.60  # the width of the bars
    figure_title = 'number of annotations per area group'

    rects = ax.bar(x, areaRngLbl2Number.values(), width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of annotations')
    ax.set_title(figure_title)
    ax.set_xticks(x)
    ax.set_xticklabels(areaRngLbl)

    # Add score texts over bars
    autolabel(ax, rects)

    # Save plot
    fig.tight_layout()
    fig.savefig(outDir + f'/{figure_title}.png')
    plt.close(fig)


def make_gt_area_histogram_plot(cocoEval, outDir):
    n_bins = 100
    areas = [ann['area'] for ann in cocoEval.cocoGt.anns.values()]

    # init figure
    figure_title = 'gt annotation areas histogram plot'
    fig, ax = plt.subplots()

    # Set the number of bins
    ax.hist(np.sqrt(areas), bins=n_bins)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Squareroot Area')
    ax.set_ylabel('Number of annotations')
    ax.set_title(figure_title)

    # Save plot
    fig.tight_layout()
    fig.savefig(outDir + f'/{figure_title}.png')
    plt.close(fig)


def analyze_individual_category(k,
                                cocoDt,
                                cocoGt,
                                catId,
                                iou_type,
                                areas=None):
    nm = cocoGt.loadCats(catId)[0]
    print(f'--------------analyzing {k + 1}-{nm["name"]}---------------')
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    child_catIds = gt.getCatIds(supNms=[nm['supercategory']])
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] in child_catIds and ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_supercategory'] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return k, ps_


def analyze_results(res_file,
                    ann_file,
                    res_types,
                    out_dir,
                    extraplots=None,
                    areas=None):
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']
    if areas:
        assert len(areas) == 3, '3 integers should be specified as areas, \
            representing 3 area regions'

    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print(f'-------------create {out_dir}-----------------')
        os.makedirs(directory)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        res_out_dir = out_dir + '/' + res_type + '/'
        res_directory = os.path.dirname(res_out_dir)
        if not os.path.exists(res_directory):
            print(f'-------------create {res_out_dir}-----------------')
            os.makedirs(res_directory)
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
        cocoEval.params.maxDets = [100]
        if areas:
            cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                       [areas[0], areas[1]],
                                       [areas[1], areas[2]]]
        cocoEval.evaluate()
        cocoEval.accumulate()
        ps = cocoEval.eval['precision']
        ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])
        catIds = cocoGt.getCatIds()
        recThrs = cocoEval.params.recThrs
        with Pool(processes=48) as pool:
            args = [(k, cocoDt, cocoGt, catId, iou_type, areas)
                    for k, catId in enumerate(catIds)]
            analyze_results = pool.starmap(analyze_individual_category, args)
        for k, catId in enumerate(catIds):
            nm = cocoGt.loadCats(catId)[0]
            print(f'--------------saving {k + 1}-{nm["name"]}---------------')
            analyze_result = analyze_results[k]
            assert k == analyze_result[0]
            ps_supercategory = analyze_result[1]['ps_supercategory']
            ps_allcategory = analyze_result[1]['ps_allcategory']
            # compute precision but ignore superclass confusion
            ps[3, :, k, :, :] = ps_supercategory
            # compute precision but ignore any class confusion
            ps[4, :, k, :, :] = ps_allcategory
            # fill in background and false negative errors and plot
            ps[ps == -1] = 0
            ps[5, :, k, :, :] = ps[4, :, k, :, :] > 0
            ps[6, :, k, :, :] = 1.0
            makeplot(recThrs, ps[:, :, k], res_out_dir, nm['name'], iou_type)
            if extraplots:
                makebarplot(recThrs, ps[:, :, k], res_out_dir, nm['name'],
                            iou_type)
        makeplot(recThrs, ps, res_out_dir, 'allclass', iou_type)
        if extraplots:
            makebarplot(recThrs, ps, res_out_dir, 'allclass', iou_type)
            make_gt_area_group_numbers_plot(
                cocoEval=cocoEval, outDir=res_out_dir, verbose=True)
            make_gt_area_histogram_plot(cocoEval=cocoEval, outDir=res_out_dir)


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument('out_dir', help='dir to save analyze result images')
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/instances_val2017.json',
        help='annotation file path')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        '--extraplots',
        action='store_true',
        help='export extra bar/stat plots')
    parser.add_argument(
        '--areas',
        type=int,
        nargs='+',
        default=[1024, 9216, 10000000000],
        help='area regions')
    args = parser.parse_args()
    analyze_results(
        args.result,
        args.ann,
        args.types,
        out_dir=args.out_dir,
        extraplots=args.extraplots,
        areas=args.areas)


if __name__ == '__main__':
    main()
