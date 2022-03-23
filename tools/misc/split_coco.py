# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np


def split_coco(data_root, fold, percent):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root: root of dataset
        fold: the fold of dataset and set as random seed for data split
        percent: percentage of labeled data
    """

    def save_anno(name, images, annotations):
        print(f'Starting to split data {name}.json '
              f'saved {len(images)} images and {len(annotations)}annotations')

        sub_annos = {}
        sub_annos['images'] = images
        sub_annos['annotations'] = annotations
        sub_annos['licenses'] = annos['licenses']
        sub_annos['categories'] = annos['categories']
        sub_annos['info'] = annos['info']
        # the path of semi-supervised annotations is ./data/coco_semi_annos,
        # because ./data/coco may be a soft link
        path = '/'.join(ann_file.split('/')[:-3] + ['coco_semi_annos'])
        mmcv.mkdir_or_exist(path)
        mmcv.dump(sub_annos, f'{path}/{name}.json')

        print(f'Finishing to split data {name}.json '
              f'saved {len(images)} images and {len(annotations)} annotations')

    np.random.seed(fold)
    ann_file = osp.join(data_root, 'annotations/instances_train2017.json')
    annos = mmcv.load(ann_file)

    image_list = annos['images']
    labeled_tot = int(percent / 100. * len(image_list))
    labeled_ind = set(
        np.random.choice(range(len(image_list)), size=labeled_tot))
    labeled_id, labeled_images, unlabeled_images = [], [], []

    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i])
            labeled_id.append(image_list[i]['id'])
        else:
            unlabeled_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_id = set(labeled_id)
    labeled_annotations, unlabeled_annotations = [], []

    for anno in annos['annotations']:
        if anno['image_id'] in labeled_id:
            labeled_annotations.append(anno)
        else:
            unlabeled_annotations.append(anno)

    # save labeled and unlabeled
    labeled_name = f'instances_train2017.{fold}@{percent}'
    unlabeled_name = f'instances_train2017.{fold}@{percent}-unlabeled'

    save_anno(labeled_name, labeled_images, labeled_annotations)
    save_anno(unlabeled_name, unlabeled_images, unlabeled_annotations)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='data root of coco dataset',
        default='./data/coco/')
    parser.add_argument('--k-fold', type=int, help='k-fold', default=5)
    parser.add_argument(
        '--percent',
        type=float,
        nargs='+',
        help='the percent of labeled data in train set',
        default=[1, 2, 5, 10])

    args = parser.parse_args()

    for fold in range(1, args.k_fold + 1):
        for percent in args.percent:
            split_coco(args.data_root, fold, percent)
