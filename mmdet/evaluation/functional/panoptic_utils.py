# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2018, Alexander Kirillov
# This file supports `backend_args` for `panopticapi`,
# the source code is copied from `panopticapi`,
# only the way to load the gt images is modified.
import multiprocessing
import os

import mmcv
import numpy as np
from mmengine.fileio import get

# A custom value to distinguish instance ID and category ID; need to
# be greater than the number of categories.
# For a pixel in the panoptic result map:
#   pan_id = ins_id * INSTANCE_OFFSET + cat_id
INSTANCE_OFFSET = 1000

try:
    from panopticapi.evaluation import OFFSET, VOID, PQStat
    from panopticapi.utils import rgb2id
except ImportError:
    PQStat = None
    rgb2id = None
    VOID = 0
    OFFSET = 256 * 256 * 256


def pq_compute_single_core(proc_id,
                           annotation_set,
                           gt_folder,
                           pred_folder,
                           categories,
                           backend_args=None,
                           print_log=False):
    """The single core function to evaluate the metric of Panoptic
    Segmentation.

    Same as the function with the same name in `panopticapi`. Only the function
    to load the images is changed to use the file client.

    Args:
        proc_id (int): The id of the mini process.
        gt_folder (str): The path of the ground truth images.
        pred_folder (str): The path of the prediction images.
        categories (str): The categories of the dataset.
        backend_args (object): The Backend of the dataset. If None,
            the backend will be set to `local`.
        print_log (bool): Whether to print the log. Defaults to False.
    """
    if PQStat is None:
        raise RuntimeError(
            'panopticapi is not installed, please install it by: '
            'pip install git+https://github.com/cocodataset/'
            'panopticapi.git.')

    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if print_log and idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(
                proc_id, idx, len(annotation_set)))
        idx += 1
        # The gt images can be on the local disk or `ceph`, so we use
        # backend here.
        img_bytes = get(
            os.path.join(gt_folder, gt_ann['file_name']),
            backend_args=backend_args)
        pan_gt = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
        pan_gt = rgb2id(pan_gt)

        # The predictions can only be on the local dist now.
        pan_pred = mmcv.imread(
            os.path.join(pred_folder, pred_ann['file_name']),
            flag='color',
            channel_order='rgb')
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    'In the image with ID {} segment with ID {} is '
                    'presented in PNG and not presented in JSON.'.format(
                        gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError(
                    'In the image with ID {} segment with ID {} has '
                    'unknown category_id {}.'.format(
                        gt_ann['image_id'], label,
                        pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError(
                'In the image with ID {} the following segment IDs {} '
                'are presented in JSON and not presented in PNG.'.format(
                    gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(
            np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                    'category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                'area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info['category_id']], pred_label),
                    0)
            # predicted segment is ignored if more than half of
            # the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1

    if print_log:
        print('Core: {}, all {} images processed'.format(
            proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list,
                          gt_folder,
                          pred_folder,
                          categories,
                          backend_args=None,
                          nproc=32):
    """Evaluate the metrics of Panoptic Segmentation with multithreading.

    Same as the function with the same name in `panopticapi`.

    Args:
        matched_annotations_list (list): The matched annotation list. Each
            element is a tuple of annotations of the same image with the
            format (gt_anns, pred_anns).
        gt_folder (str): The path of the ground truth images.
        pred_folder (str): The path of the prediction images.
        categories (str): The categories of the dataset.
        backend_args (object): The file client of the dataset. If None,
            the backend will be set to `local`.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When `nproc` exceeds the number of cpu cores,
            the number of cpu cores is used.
    """
    if PQStat is None:
        raise RuntimeError(
            'panopticapi is not installed, please install it by: '
            'pip install git+https://github.com/cocodataset/'
            'panopticapi.git.')

    cpu_num = min(nproc, multiprocessing.cpu_count())

    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print('Number of cores: {}, images per core: {}'.format(
        cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder,
                                 pred_folder, categories, backend_args))
        processes.append(p)

    # Close the process pool, otherwise it will lead to memory
    # leaking problems.
    workers.close()
    workers.join()

    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()

    return pq_stat
