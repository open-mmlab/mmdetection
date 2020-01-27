from bisect import bisect
from collections import namedtuple

import mmcv
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from mmdet import datasets


def voc_ap(recall, precision, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
    else:
        # Correct AP calculation.
        # First append sentinel values at the end.
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points
        # where X axis (recall) changes value.
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # And sum (\Delta recall) * prec.
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def miss_rate(miss_rates, fppis, fppi_level=0.1):
    position = bisect(fppis, fppi_level)
    p1 = position - 1
    p2 = position if position < len(miss_rates) else p1
    return 0.5 * (miss_rates[p1] + miss_rates[p2])


def evaluate_detections(ground_truth, predictions, class_name, overlap_threshold=0.5,
                        allow_multiple_matches_per_ignored=True,
                        verbose=True):
    Detection = namedtuple('Detection', ['image', 'bbox', 'score', 'gt_match'])
    GT = namedtuple('GroundTruth', ['bbox', 'is_matched', 'is_ignored'])
    detections = [Detection(image=img_pred.image_path,
                            bbox=np.array(obj_pred["bbox"]),
                            score=obj_pred.get("score", 0.0),
                            gt_match=-1)
                  for img_pred in predictions
                  for obj_pred in img_pred
                  if obj_pred["type"] == class_name]

    scores = np.array([detection.score for detection in detections])
    sorted_ind = np.argsort(-scores)
    detections = [detections[i] for i in sorted_ind]

    gts = {img_gt.image_path: GT(
        bbox=np.vstack([np.array(obj_gt["bbox"]) for obj_gt in img_gt]) if img_gt else np.empty(
            (0, 4)),
        is_matched=np.zeros(len(img_gt), dtype=bool),
        is_ignored=np.array([obj_gt.get("is_ignored", False) for obj_gt in img_gt], dtype=bool))
        for img_gt in ground_truth}

    nd = len(detections)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for i, detection in tqdm(enumerate(detections), desc="Processing detections",
                             disable=not verbose):
        image_path = detection.image
        bboxes_gt = gts[image_path].bbox
        bbox = detection.bbox
        max_overlap = -np.inf

        if bboxes_gt is not None and len(bboxes_gt) > 0:
            intersection_xmin = np.maximum(bboxes_gt[:, 0], bbox[0])
            intersection_ymin = np.maximum(bboxes_gt[:, 1], bbox[1])
            intersection_xmax = np.minimum(bboxes_gt[:, 0] + bboxes_gt[:, 2], bbox[0] + bbox[2])
            intersection_ymax = np.minimum(bboxes_gt[:, 1] + bboxes_gt[:, 3], bbox[1] + bbox[3])
            intersection_width = np.maximum(intersection_xmax - intersection_xmin, 0.)
            intersection_height = np.maximum(intersection_ymax - intersection_ymin, 0.)
            intersection = intersection_width * intersection_height

            det_area = bbox[2] * bbox[3]
            gt_area = bboxes_gt[:, 2] * bboxes_gt[:, 3]
            union = (det_area + gt_area - intersection)
            ignored_mask = gts[image_path].is_ignored
            if allow_multiple_matches_per_ignored:
                if np.any(ignored_mask):
                    union[ignored_mask] = det_area

            overlaps = intersection / union
            # Match not ignored ground truths first.
            if np.any(~ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            # If match with non-ignored ground truth is not good enough,
            # try to match with ignored ones.
            if max_overlap < overlap_threshold and np.any(ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[~ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            detections[i] = detection._replace(gt_match=argmax_overlap)

        if max_overlap >= overlap_threshold:
            if not gts[image_path].is_ignored[argmax_overlap]:
                if not gts[image_path].is_matched[argmax_overlap]:
                    tp[i] = 1.
                    gts[image_path].is_matched[argmax_overlap] = True
                else:
                    fp[i] = 1.
            elif not allow_multiple_matches_per_ignored:
                gts[image_path].is_matched[argmax_overlap] = True
        else:
            fp[i] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    debug_visualization = False
    if debug_visualization:
        for im, bboxes_gt in gts.iteritems():
            import cv2
            print(im)
            image = cv2.imread(im)
            image_gt = np.copy(image)
            for b in bboxes_gt.bbox:
                cv2.rectangle(image_gt, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(255, 255, 0),
                              thickness=2)
            cv2.imshow("gt", image_gt)
            for detection in detections:
                if detection.image != im:
                    continue
                b = detection.bbox
                cv2.rectangle(image, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(0, 255, 0),
                              thickness=2)
                if detection.gt_match is not None:
                    b = bboxes_gt.bbox[detection.gt_match]
                    cv2.rectangle(image, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(0, 0, 255),
                                  thickness=1)
                cv2.imshow("image", image)
                cv2.waitKey(0)

    # Handle equal-score detections.
    # Get index of the last occurrence of a score.
    ind = len(scores) - np.unique(scores[sorted_ind[::-1]], return_index=True)[1] - 1
    ind = ind[::-1]
    # Though away redundant points.
    fp = fp[ind]
    tp = tp[ind]

    total_positives_num = np.sum([np.count_nonzero(~gt.is_ignored) for gt in gts.values()])
    recall = tp / float(total_positives_num)
    # Avoid divide by zero in case the first detection matches an ignored ground truth.
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    miss_rate = 1.0 - recall
    fppi = fp / float(len(gts))

    return recall, precision, miss_rate, fppi


class ImageAnnotation:
    def __init__(self, image_path, objects=None, ignore_regs=None):
        self.image_path = image_path
        self.objects = objects if objects else []
        self.ignore_regs = ignore_regs if ignore_regs else []

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]


def points_2_xywh(box):
    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    box = [int(round(x)) for x in box]
    return box


def clip_bbox(bbox, im_size):
    bbox = np.maximum(np.copy(bbox), 0)
    x, y, w, h = bbox
    w = min(x + w, im_size[0]) - x
    h = min(y + h, im_size[1]) - y
    if w == 0 and h == 0:
        x = y = w = h = -1
    return np.array([x, y, w, h])


def voc_eval(result_file, dataset, iou_thr, image_size, obj_size):
    det_results = mmcv.load(result_file)
    groundtruth = []
    predictions = []

    MIN_DETECTION_CONFIDENCE = 0.01

    for i, _ in enumerate(tqdm(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']

        # +1 is to compensate pre-processing in XMLDataset
        if isinstance(dataset, datasets.XMLDataset):
            bboxes = [np.array(bbox) + 1 for bbox in bboxes]
        # convert from [xmin, ymin, xmax, ymax] to [xmin, ymin, w, h]
        bboxes = [points_2_xywh(bbox) for bbox in bboxes]
        # clip bboxes
        bboxes = [clip_bbox(bbox, image_size) for bbox in bboxes]
        # filter out boxes with to small height or with invalid size (-1)
        ignored = [not (obj_size[0] <= b[3] <= obj_size[1]) or np.any(b == -1) for b in bboxes]
        objects = [{'bbox': bbox, 'is_ignored': ignor} for bbox, ignor in zip(bboxes, ignored)]
        groundtruth.append(ImageAnnotation(dataset.img_infos[i]['id'], objects))

        # filter out predictions with too low confidence
        detections = [{'bbox': points_2_xywh(bbox[:4]), 'score': bbox[4], 'type': 'face'} for bbox
                      in det_results[i][0] if bbox[4] > MIN_DETECTION_CONFIDENCE]
        predictions.append(ImageAnnotation(dataset.img_infos[i]['id'], detections))

    recall, precision, miss_rates, fppis = evaluate_detections(groundtruth,
                                                               predictions,
                                                               'face',
                                                               allow_multiple_matches_per_ignored=
                                                               True,
                                                               overlap_threshold=iou_thr)

    mr = miss_rate(miss_rates, fppis) * 100
    ap = voc_ap(recall, precision) * 100

    print(f'ImageSize = {image_size}, ObjSize = {obj_size}, AP = {ap:.2f}%, MissRate = {mr:.2f}%')


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--imsize', nargs=2, type=int, default=(1024, 1024),
                        help='Image resolution. Used for filtering.')
    parser.add_argument('--objsize', nargs=2, type=int, required=True,
                        help='Viable object height range. Used for filtering.')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for evaluation')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr, args.imsize, args.objsize)


if __name__ == '__main__':
    main()
