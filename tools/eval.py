from argparse import ArgumentParser
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def generate_area_range(splitRng=32, stop_size=128):
    areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
    start = 0
    while start < stop_size:
        end = start + splitRng
        areaRng.append([start * start, end * end])
        start = end
    areaRng.append([start * start, 1e5**2])
    return areaRng


def print_summarize(iouThr=None,
                    iouThrs=None,
                    precision=None,
                    recall=None,
                    areaRng_id=4,
                    areaRngs=None,
                    maxDets_id=2,
                    maxDets=None):
    assert (precision is not None) or (recall is not None)
    iStr = ' {:<18} {} @[ IoU={:<9} | size={:>5}-{:>5} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if precision is not None else 'Average Recall'
    typeStr = '(AP)' if precision is not None else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(iouThrs[0], iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [areaRng_id]
    mind = [maxDets_id]
    if precision is not None:
        # dimension of precision: [TxRxKxAxM]
        s = precision
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = recall
        if iouThr is not None:
            t = np.where(iouThr == iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(
        iStr.format(
            titleStr, typeStr, iouStr, np.sqrt(areaRngs[areaRng_id][0]),
            np.sqrt(areaRngs[areaRng_id][1])
            if np.sqrt(areaRngs[areaRng_id][1]) < 999 else 'max',
            maxDets[maxDets_id], mean_s))


def eval_results(res_file, ann_file, res_types, splitRng):
    for res_type in res_types:
        assert res_type in ['proposal', 'bbox', 'segm', 'keypoints']

    areaRng = generate_area_range(splitRng)
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = imgIds
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = [100, 300, 1000]
        cocoEval.params.areaRng = areaRng
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ps = cocoEval.eval['precision']
        rc = cocoEval.eval['recall']
        for i in range(len(areaRng)):
            print_summarize(None, cocoEval.params.iouThrs, ps, None, i,
                            areaRng, 2, cocoEval.params.maxDets)


def makeplot(rs, ps, outDir, class_name):
    cs = np.vstack([
        np.ones((2, 3)),
        np.array([.31, .51, .74]),
        np.array([.75, .31, .30]),
        np.array([.36, .90, .38]),
        np.array([.50, .39, .64]),
        np.array([1, .6, 0])
    ])
    areaNames = ['all', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_tile = class_name + '-' + areaNames[i]
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
            makeplot(recThrs, ps[:, :, k], out_dir, nm['name'])
        makeplot(recThrs, ps, out_dir, 'all')


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument(
        '--ann',
        default='/mnt/SSD/dataset/coco/annotations/instances_minival2017.json',
        help='annotation file path')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        '--analyze', action='store_true', help='whether to analyze results')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='dir to save analyze result images')
    parser.add_argument(
        '--splitRng',
        type=int,
        default=32,
        help='range to split area in evaluation')
    args = parser.parse_args()
    if not args.analyze:
        eval_results(args.result, args.ann, args.types, splitRng=args.splitRng)
    else:
        assert args.out_dir is not None
        analyze_results(
            args.result, args.ann, args.types, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
