# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import mmengine
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='CrowdHuman to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of CrowdHuman annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def load_odgt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_infos = [json.loads(line.strip('\n')) for line in lines]
    return data_infos


def convert_crowdhuman(ann_dir, save_dir, mode='train'):
    """Convert CrowdHuman dataset in COCO style.

    Args:
        ann_dir (str): The path of CrowdHuman dataset.
        save_dir (str): The path to save annotation files.
        mode (str): Convert train dataset or validation dataset. Options are
            'train', 'val'. Default: 'train'.
    """
    assert mode in ['train', 'val']

    records = dict(img_id=1, ann_id=1)
    outputs = defaultdict(list)
    outputs['categories'] = [dict(id=1, name='pedestrian')]

    data_infos = load_odgt(osp.join(ann_dir, f'annotation_{mode}.odgt'))
    for data_info in tqdm(data_infos):
        img_name = osp.join('Images', f"{data_info['ID']}.jpg")
        img = Image.open(osp.join(ann_dir, mode, img_name))
        width, height = img.size[:2]
        image = dict(
            file_name=img_name,
            height=height,
            width=width,
            id=records['img_id'])
        outputs['images'].append(image)

        if mode != 'test':
            for ann_info in data_info['gtboxes']:
                bbox = ann_info['fbox']
                if 'extra' in ann_info and 'ignore' in ann_info[
                        'extra'] and ann_info['extra']['ignore'] == 1:
                    iscrowd = True
                else:
                    iscrowd = False
                ann = dict(
                    id=records['ann_id'],
                    image_id=records['img_id'],
                    category_id=outputs['categories'][0]['id'],
                    vis_bbox=ann_info['vbox'],
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    iscrowd=iscrowd)
                outputs['annotations'].append(ann)
                records['ann_id'] += 1
        records['img_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmengine.dump(outputs, osp.join(save_dir, f'crowdhuman_{mode}.json'))
    print(f'-----CrowdHuman {mode} set------')
    print(f'total {records["img_id"] - 1} images')
    if mode != 'test':
        print(f'{records["ann_id"] - 1} pedestrians are annotated.')
    print('-----------------------')


def main():
    args = parse_args()
    convert_crowdhuman(args.input, args.output, mode='train')
    convert_crowdhuman(args.input, args.output, mode='val')


if __name__ == '__main__':
    main()
