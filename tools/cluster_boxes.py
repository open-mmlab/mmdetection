# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import json
import logging
import os

import mmcv
import numpy as np
from mmdet.datasets import build_dataloader, build_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument('--config', type=str, help='Training configuration.')
    input.add_argument('--coco_annotation', type=str,
                       help='COCO annotation. This variant is much faster than --config in case of '
                            'COCO annotation.')
    parser.add_argument('--root', help='Images root folder.')
    parser.add_argument('--image_size_wh', nargs=2, type=int)
    parser.add_argument('--n_clust', type=int, required=True)
    parser.add_argument('--min_box_size', help='min bbox Width and Height', nargs=2, type=int,
                        default=(0, 0))
    args = parser.parse_args()
    return args


def get_sizes_from_config(config_path, min_box_size):
    cfg = mmcv.Config.fromfile(config_path)

    dataset = build_dataset(cfg.data.train)
    logging.info(dataset)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    logging.info('Collecting statistics...')
    wh_stats = []
    for data_batch in tqdm(iter(data_loader)):
        boxes = data_batch['gt_bboxes'].data[0][0].numpy()
        for box in boxes:
            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            if w > min_box_size[0] and h > min_box_size[1]:
                wh_stats.append((w, h))

    return wh_stats


def get_sizes_from_coco(annotation_path, root, target_image_wh, min_box_size):
    import imagesize
    with open(annotation_path) as f:
        content = json.load(f)

    images_wh = {}
    wh_stats = []
    for image_info in tqdm(content['images']):
        image_path = os.path.join(root, image_info['file_name'])
        images_wh[image_info['id']] = imagesize.get(image_path)

    for ann in content['annotations']:
        w, h = ann['bbox'][2:4]
        image_wh = images_wh[ann['image_id']]
        w, h = w / image_wh[0], h / image_wh[1]
        w, h = w * target_image_wh[0], h * target_image_wh[1]
        if w > min_box_size[0] and h > min_box_size[1]:
            wh_stats.append((w, h))

    return wh_stats


def main(args):
    assert args.config or args.coco_annotation

    if args.config:
        assert not args.image_size_wh
        assert not args.root
        wh_stats = get_sizes_from_config(args.config, args.min_box_size)

    if args.coco_annotation:
        assert args.image_size_wh
        assert args.root
        wh_stats = get_sizes_from_coco(args.coco_annotation, args.root, args.image_size_wh,
                                       args.min_box_size)

    kmeans = KMeans(init='k-means++', n_clusters=args.n_clust, random_state=0).fit(wh_stats)
    centers = kmeans.cluster_centers_

    areas = np.sqrt([c[0] * c[1] for c in centers])
    idx = np.argsort(areas)

    for i in idx:
        center = centers[i]
        logging.info('width: {:.3f}'.format(center[0]))
        logging.info('height: {:.3f}'.format(center[1]))

    widths = [centers[i][0] for i in idx]
    heights = [centers[i][1] for i in idx]
    logging.info(widths)
    logging.info(heights)


if __name__ == '__main__':
    log_format = '{levelname} {asctime} {filename}:{lineno:>4d}] {message}'
    date_format = '%d-%m-%y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format, style='{')
    args = parse_args()
    main(args)
