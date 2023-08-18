# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample


def weighted_boxes_fusion(inference_results: list,
                          img_shape: tuple,
                          weights: list = None,
                          iou_thr: float = 0.55,
                          skip_box_thr: float = 0.0,
                          conf_type: str = 'avg',
                          allows_overflow: bool = False) -> DetDataSample:
    """weighted boxes fusion (<https://arxiv.org/abs/1910.13302> or <https://ww
    w.sciencedirect.com/science/article/abs/pii/S0262885621000226>) is a method
    for fusing predictions from different object detection models, which
    utilizes confidence scores of all proposed bounding boxes to construct
    averaged boxes.

    Args:
        inference_results (list): list of DetDataSample which are results
            produced by several models.
        img_shape: image shape, (h, w, 3)
        weights: list of weights for each model.
                Default: None, which means weight == 1 for each model
        iou_thr: IoU value for boxes to be a match
        skip_box_thr: exclude boxes with score lower than this variable.
        conf_type: how to calculate confidence in weighted boxes.
            'avg': average value,
            'max': maximum value,
            'box_and_model_avg': box and model wise hybrid weighted average,
            'absent_model_aware_avg': weighted average that takes into
                            account the absent model.
        allows_overflow: false if we want confidence score not exceed 1.0.

    Returns:
        DetDataSample: fusion result
    """

    boxes_list = []
    scores_list = []
    labels_list = []

    for model_result in inference_results:
        boxes_list.append(model_result.pred_instances.bboxes)
        scores_list.append(model_result.pred_instances.scores)
        labels_list.append(model_result.pred_instances.labels)

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: '
              '{}. Set weights equal to 1.'.format(
                  len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in [
            'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    ]:
        print('Unknown conf_type: {}. Must be "avg", '
              '"max" or "box_and_model_avg", '
              'or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list,
                                     img_shape, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return DetDataSample()

    overall_boxes = []

    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j],
                                                     iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(
                    new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(
                    clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[
                    idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[
                    i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (
                        weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(
                    len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(
                    clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]

    bboxes = torch.Tensor(overall_boxes[:, 4:])
    scores = torch.Tensor(overall_boxes[:, 1])
    labels = torch.Tensor(overall_boxes[:, 0]).int()

    pred_instances = InstanceData()
    pred_instances.bboxes = bboxes
    pred_instances.scores = scores
    pred_instances.labels = labels

    fusion_result = DetDataSample(pred_instances=pred_instances)

    return fusion_result


def prefilter_boxes(boxes, scores, labels, img_shape, weights, thr):

    new_boxes = dict()

    h = float(img_shape[0])
    w = float(img_shape[1])

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to '
                  'length of scores array: {} != {}'.format(
                      len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to '
                  'length of labels array: {} != {}'.format(
                      len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > w:
                warnings.warn(
                    'X1 > image_width in box. Set it to image_width.')
                x1 = w
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > w:
                warnings.warn(
                    'X2 > image_width in box. Set it to image_width.')
                x2 = w
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > h:
                warnings.warn(
                    'Y1 > image_heigth in box. Set it to image_heigth.')
                y1 = h
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > h:
                warnings.warn(
                    'Y2 > image_heigth in box. Set it to image_heigth.')
                y2 = h
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn('Zero area box skipped: {}.'.format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [
                int(label),
                float(score) * weights[t], weights[t], t, x1, y1, x2, y2
            ]

            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1
    box[4:] /= conf

    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):

    def bb_iou_array(boxes, new_box):
        # bb intersection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou
