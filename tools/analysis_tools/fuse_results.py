import argparse

from mmengine.fileio import dump, load
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.models.utils import weighted_boxes_fusion


def parse_args():
    parser = argparse.ArgumentParser(description='Fusion image \
        prediction results using Weighted \
        Boxes Fusion from multiple models.')
    parser.add_argument(
        'pred-results',
        type=str,
        nargs='+',
        help='files of prediction results \
                    from multiple models, json format.')
    parser.add_argument('--annotation', type=str, help='annotation file path')
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
        help='IoU value for boxes to be a match in wbf.')
    parser.add_argument(
        '--skip-box-thr',
        type=float,
        default=0.0,
        help='exclude boxes with score lower than this variable in wbf.')
    parser.add_argument(
        '--conf-type',
        type=str,
        default='avg',
        help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--eval-single',
        action='store_true',
        help='whether evaluate each single model result.')
    parser.add_argument(
        '--save-fusion-results',
        action='store_true',
        help='whether save fusion result')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    assert len(args.models_name) == len(args.pred_results), \
        'the quantities of model names and prediction results are not equal'

    cocoGT = COCO(args.annotation)

    predicts_raw = []

    models_name = ['model_' + str(i) for i in range(len(args.pred_results))]

    for model_name, path in \
            zip(models_name, args.pred_results):
        pred = load(path)
        predicts_raw.append(pred)

        if args.eval_single:
            print_log(f'Evaluate {model_name}...')
            cocoDt = cocoGT.loadRes(pred)
            coco_eval = COCOeval(cocoGT, cocoDt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    predict = {
        str(image_id): {
            'bboxes_list': [[] for _ in range(len(predicts_raw))],
            'scores_list': [[] for _ in range(len(predicts_raw))],
            'labels_list': [[] for _ in range(len(predicts_raw))]
        }
        for image_id in cocoGT.getImgIds()
    }

    for i, pred_single in enumerate(predicts_raw):
        for pred in pred_single:
            p = predict[str(pred['image_id'])]
            p['bboxes_list'][i].append(pred['bbox'])
            p['scores_list'][i].append(pred['score'])
            p['labels_list'][i].append(pred['category_id'])

    result = []
    prog_bar = ProgressBar(len(predict))
    for image_id, res in predict.items():
        bboxes, scores, labels = weighted_boxes_fusion(
            res['bboxes_list'],
            res['scores_list'],
            res['labels_list'],
            weights=args.weights,
            iou_thr=args.fusion_iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type=args.conf_type)

        for bbox, score, label in zip(bboxes, scores, labels):
            result.append({
                'bbox': bbox.numpy().tolist(),
                'category_id': int(label),
                'image_id': int(image_id),
                'score': float(score)
            })

        prog_bar.update()

    if args.save_fusion_results:
        out_file = args.out_dir + '/fusion_results.json'
        dump(result, file=out_file)
        print_log(
            f'Fusion results have been saved to {out_file}.', logger='current')

    print_log('Evaluate fusion results using wbf...')
    cocoDt = cocoGT.loadRes(result)
    coco_eval = COCOeval(cocoGT, cocoDt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
