# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import random
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json', type=str, required=True, help='COCO json label path')
    parser.add_argument(
        '--out-dir', type=str, required=True, help='output path')
    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        help='ratio for sub dataset, if set 2 number then will generate '
        'trainval + test (eg. "0.8 0.1 0.1" or "2 1 1"), if set 3 number '
        'then will generate train + val + test (eg. "0.85 0.15" or "2 1")')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether to display in disorder')
    parser.add_argument('--seed', default=-1, type=int, help='seed')
    args = parser.parse_args()
    return args


def split_coco_dataset(coco_json_path: str, save_dir: str, ratios: list,
                       shuffle: bool, seed: int):
    if not Path(coco_json_path).exists():
        raise FileNotFoundError(f'Can not not found {coco_json_path}')

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    # ratio normalize
    ratios = np.array(ratios) / np.array(ratios).sum()

    if len(ratios) == 2:
        ratio_train, ratio_test = ratios
        ratio_val = 0
        train_type = 'trainval'
    elif len(ratios) == 3:
        ratio_train, ratio_val, ratio_test = ratios
        train_type = 'train'
    else:
        raise ValueError('ratios must set 2 or 3 group!')

    # Read coco info
    coco = COCO(coco_json_path)
    coco_image_ids = coco.getImgIds()

    # gen image number of each dataset
    val_image_num = int(len(coco_image_ids) * ratio_val)
    test_image_num = int(len(coco_image_ids) * ratio_test)
    train_image_num = len(coco_image_ids) - val_image_num - test_image_num
    print('Split info: ====== \n'
          f'Train ratio = {ratio_train}, number = {train_image_num}\n'
          f'Val ratio = {ratio_val}, number = {val_image_num}\n'
          f'Test ratio = {ratio_test}, number = {test_image_num}')

    seed = int(seed)
    if seed != -1:
        print(f'Set the global seed: {seed}')
        np.random.seed(seed)

    if shuffle:
        print('shuffle dataset.')
        random.shuffle(coco_image_ids)

    # split each dataset
    train_image_ids = coco_image_ids[:train_image_num]
    if val_image_num != 0:
        val_image_ids = coco_image_ids[train_image_num:train_image_num +
                                       val_image_num]
    else:
        val_image_ids = None
    test_image_ids = coco_image_ids[train_image_num + val_image_num:]

    # Save new json
    categories = coco.loadCats(coco.getCatIds())
    for img_id_list in [train_image_ids, val_image_ids, test_image_ids]:
        if img_id_list is None:
            continue

        # Gen new json
        img_dict = {
            'images': coco.loadImgs(ids=img_id_list),
            'categories': categories,
            'annotations': coco.loadAnns(coco.getAnnIds(imgIds=img_id_list))
        }

        # save json
        if img_id_list == train_image_ids:
            json_file_path = Path(save_dir, f'{train_type}.json')
        elif img_id_list == val_image_ids:
            json_file_path = Path(save_dir, 'val.json')
        elif img_id_list == test_image_ids:
            json_file_path = Path(save_dir, 'test.json')
        else:
            raise ValueError('img_id_list ERROR!')

        print(f'Saving json to {json_file_path}')
        with open(json_file_path, 'w') as f_json:
            json.dump(img_dict, f_json, ensure_ascii=False, indent=2)

    print('All done!')


def main():
    args = parse_args()
    split_coco_dataset(args.json, args.out_dir, args.ratios, args.shuffle,
                       args.seed)


if __name__ == '__main__':
    main()