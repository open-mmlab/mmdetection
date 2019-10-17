import copy
from argparse import ArgumentParser
from multiprocessing import Pool

import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.core import coco_eval


def coco_eval_individual_category(k, cocoDt, cocoGt, catId, iou_type):
    nm = cocoGt.loadCats(catId)[0]
    print('--------------evaluating {}-{}---------------'.format(
        k + 1, nm['name']))
    ap_ = {}
    dt = copy.deepcopy(cocoDt)
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute mAP but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap_['ap_allcategory'] = cocoEval.stats
    return k, ap_


def coco_eval_class_wise(result_files,
                         result_types,
                         coco,
                         max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'bbox', 'segm', 'keypoints'
        ], "Currently not support {} in class wise evaluation".format(res_type)

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        if isinstance(result_files, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            assert TypeError('result_files must be a str or dict')
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        iou_type = 'bbox' if res_type == 'proposal' else res_type

        # eval class_wise mAP
        catIds = coco.getCatIds()
        with Pool(processes=48) as pool:
            args = [(k, coco_dets, coco, catId, iou_type)
                    for k, catId in enumerate(catIds)]
            analyze_results = pool.starmap(coco_eval_individual_category, args)

        # show ap for each class
        for k, catId in enumerate(catIds):
            nm = coco.loadCats(catId)[0]
            analyze_result = analyze_results[k]
            assert k == analyze_result[0]
            ap_allcategory = analyze_result[1]['ap_allcategory']
            ap_str = ''
            for ap in ap_allcategory:
                ap_str = ap_str + '{:0.3f} '.format(ap)
            print("{:02d}-{:<15}: {}".format(k + 1, nm['name'], ap_str))


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    parser.add_argument(
        '--class_wise', action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    if args.class_wise:
        coco_eval_class_wise(args.result, args.types, args.ann, args.max_dets)
    else:
        coco_eval(args.result, args.types, args.ann, args.max_dets)


if __name__ == '__main__':
    main()
