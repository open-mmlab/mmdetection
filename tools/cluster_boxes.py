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
import os

import mmcv
import numpy as np
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import ExtendedDictAction
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
    parser.add_argument('--image_size_wh', nargs=2, type=int, default=(256, 256))
    parser.add_argument('--n_clust', type=int, required=True)
    parser.add_argument('--min_box_size', help='min bbox Width and Height', nargs=2, type=int,
                        default=(0, 0))
    parser.add_argument('--group_as', type=int, nargs='+',
                        help='If it is defined clustered widths and heights will be grouped by '
                             'numbers specified here.')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')
    parser.add_argument('--out')
    return parser.parse_args()


def get_sizes_from_config(config_path, target_image_wh, min_box_size, update_config):
    cfg = mmcv.Config.fromfile(config_path)
    if update_config is not None:
        cfg.merge_from_dict(update_config)

    if cfg.data.train.dataset.type == 'CocoDataset':
        annotation_path = cfg.data.train.dataset.ann_file
        root = cfg.data.train.dataset.img_prefix
        return get_sizes_from_coco(annotation_path, root, target_image_wh, min_box_size)

    dataset = build_dataset(cfg.data.train)
    print(dataset)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    print('Collecting statistics...')
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


def print_normalized(values, size, measure):
    if isinstance(values[0], list):
        text = '[\n'
        for v in values:
            text += f' [image_{measure} * x for x in {[x / size for x in v]}],\n'
        text += ']'
    else:
        text = f'[image_{measure} * x for x in {[x / size for x in values]}]'
    print(f'normalized {measure}s')
    print(text)


def main(args):
    assert args.config or args.coco_annotation

    if args.group_as:
        assert sum(args.group_as) == args.n_clust

    if args.config:
        assert not args.root
        wh_stats = get_sizes_from_config(args.config, args.image_size_wh, args.min_box_size, args.update_config)

    if args.coco_annotation:
        assert args.root
        wh_stats = get_sizes_from_coco(args.coco_annotation, args.root, args.image_size_wh,
                                       args.min_box_size)

    kmeans = KMeans(init='k-means++', n_clusters=args.n_clust, random_state=0).fit(wh_stats)
    centers = kmeans.cluster_centers_

    areas = np.sqrt([c[0] * c[1] for c in centers])
    idx = np.argsort(areas)

    for i in idx:
        center = centers[i]
        print('width: {:.3f}'.format(center[0]))
        print('height: {:.3f}'.format(center[1]))
    print('')

    widths = [centers[i][0] for i in idx]
    heights = [centers[i][1] for i in idx]

    if args.group_as:
        group_as = np.cumsum([0] + args.group_as)
        widths = [[widths[i] for i in range(group_as[j], group_as[j + 1])] for j in
                  range(len(group_as) - 1)]
        heights = [[heights[i] for i in range(group_as[j], group_as[j + 1])] for j in
                   range(len(group_as) - 1)]

    print('widths\n', widths)
    print('heights\n', heights)
    print('')

    print_normalized(widths, args.image_size_wh[0], 'width')
    print_normalized(heights, args.image_size_wh[1], 'height')

    if args.out:
        with open(args.out, 'w') as dst_file:
            json.dump({'widths': widths, 'heights': heights}, dst_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
