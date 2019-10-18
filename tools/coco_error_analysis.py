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
        np.array([.31, .51, .74]),
        np.array([.75, .31, .30]),
        np.array([.36, .90, .38]),
        np.array([.50, .39, .64]),
        np.array([1, .6, 0])
    ])
    areaNames = ['allarea', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_tile = iou_type + '-' + class_name + '-' + areaNames[i]
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
                label=str('[{:.3f}'.format(aps[k]) + ']' + types[k]))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        plt.title(figure_tile)
        plt.legend()
        # plt.show()
        fig.savefig(outDir + '/{}.png'.format(figure_tile))
        plt.close(fig)


def analyze_individual_category(k, cocoDt, cocoGt, catId, iou_type):
    nm = cocoGt.loadCats(catId)[0]
    print('--------------analyzing {}-{}---------------'.format(
        k + 1, nm['name']))
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
        if (ann['category_id'] in child_catIds
                and ann['category_id'] != catId):
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
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
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return k, ps_


def analyze_results(res_file, ann_file, res_types, out_dir):
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']

    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print('-------------create {}-----------------'.format(out_dir))
        os.makedirs(directory)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        res_out_dir = out_dir + '/' + res_type + '/'
        res_directory = os.path.dirname(res_out_dir)
        if not os.path.exists(res_directory):
            print(
                '-------------create {}-----------------'.format(res_out_dir))
            os.makedirs(res_directory)
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [.75, .5, .1]
        cocoEval.params.maxDets = [100]
        cocoEval.evaluate()
        cocoEval.accumulate()
        ps = cocoEval.eval['precision']
        ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])
        catIds = cocoGt.getCatIds()
        recThrs = cocoEval.params.recThrs
        with Pool(processes=48) as pool:
            args = [(k, cocoDt, cocoGt, catId, iou_type)
                    for k, catId in enumerate(catIds)]
            analyze_results = pool.starmap(analyze_individual_category, args)
        for k, catId in enumerate(catIds):
            nm = cocoGt.loadCats(catId)[0]
            print('--------------saving {}-{}---------------'.format(
                k + 1, nm['name']))
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
            ps[5, :, k, :, :] = (ps[4, :, k, :, :] > 0)
            ps[6, :, k, :, :] = 1.0
            makeplot(recThrs, ps[:, :, k], res_out_dir, nm['name'], iou_type)
        makeplot(recThrs, ps, res_out_dir, 'allclass', iou_type)


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
    args = parser.parse_args()
    analyze_results(args.result, args.ann, args.types, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
