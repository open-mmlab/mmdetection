# Copyright (c) OpenMMLab. All rights reserved.
"""Get image shape on CrowdHuman dataset.

Here is an example to run this script.

Example:
    python tools/misc/get_crowdhuman_id_hw.py ${CONFIG} \
    --dataset ${DATASET_TYPE}
"""
import argparse
import json
import logging
import os.path as osp
from multiprocessing import Pool

import mmcv
from mmengine.config import Config
from mmengine.fileio import dump, get, get_text
from mmengine.logging import print_log


def parse_args():
    parser = argparse.ArgumentParser(description='Collect image metas')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--dataset',
        choices=['train', 'val'],
        help='Collect image metas from which dataset')
    parser.add_argument(
        '--nproc',
        default=10,
        type=int,
        help='Processes used for get image metas')
    args = parser.parse_args()
    return args


def get_image_metas(anno_str, img_prefix):
    id_hw = {}
    anno_dict = json.loads(anno_str)
    img_path = osp.join(img_prefix, f"{anno_dict['ID']}.jpg")
    img_id = anno_dict['ID']
    img_bytes = get(img_path)
    img = mmcv.imfrombytes(img_bytes, backend='cv2')
    id_hw[img_id] = img.shape[:2]
    return id_hw


def main():
    args = parse_args()

    # get ann_file and img_prefix from config files
    cfg = Config.fromfile(args.config)
    dataset = args.dataset
    dataloader_cfg = cfg.get(f'{dataset}_dataloader')
    ann_file = osp.join(dataloader_cfg.dataset.data_root,
                        dataloader_cfg.dataset.ann_file)
    img_prefix = osp.join(dataloader_cfg.dataset.data_root,
                          dataloader_cfg.dataset.data_prefix['img'])

    # load image metas
    print_log(
        f'loading CrowdHuman {dataset} annotation...', level=logging.INFO)
    anno_strs = get_text(ann_file).strip().split('\n')
    pool = Pool(args.nproc)
    # get image metas with multiple processes
    id_hw_temp = pool.starmap(
        get_image_metas,
        zip(anno_strs, [img_prefix for _ in range(len(anno_strs))]),
    )
    pool.close()

    # save image metas
    id_hw = {}
    for sub_dict in id_hw_temp:
        id_hw.update(sub_dict)

    data_root = osp.dirname(ann_file)
    save_path = osp.join(data_root, f'id_hw_{dataset}.json')
    print_log(
        f'\nsaving "id_hw_{dataset}.json" in "{data_root}"',
        level=logging.INFO)
    dump(id_hw, save_path, file_format='json')


if __name__ == '__main__':
    main()
