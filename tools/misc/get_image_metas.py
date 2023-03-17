# Copyright (c) OpenMMLab. All rights reserved.
"""Get image metas on a specific dataset.

Here is an example to run this script.

Example:
    python tools/misc/get_image_metas.py ${CONFIG} \
    --out ${OUTPUT FILE NAME}
"""
import argparse
import csv
import os.path as osp
from multiprocessing import Pool

import mmcv
from mmengine.config import Config
from mmengine.fileio import dump, get


def parse_args():
    parser = argparse.ArgumentParser(description='Collect image metas')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--dataset',
        default='val',
        choices=['train', 'val', 'test'],
        help='Collect image metas from which dataset')
    parser.add_argument(
        '--out',
        default='validation-image-metas.pkl',
        help='The output image metas file name. The save dir is in the '
        'same directory as `dataset.ann_file` path')
    parser.add_argument(
        '--nproc',
        default=4,
        type=int,
        help='Processes used for get image metas')
    args = parser.parse_args()
    return args


def get_metas_from_csv_style_ann_file(ann_file):
    data_infos = []
    cp_filename = None
    with open(ann_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            img_id = line[0]
            filename = f'{img_id}.jpg'
            if filename != cp_filename:
                data_infos.append(dict(filename=filename))
                cp_filename = filename
    return data_infos


def get_metas_from_txt_style_ann_file(ann_file):
    with open(ann_file) as f:
        lines = f.readlines()
    i = 0
    data_infos = []
    while i < len(lines):
        filename = lines[i].rstrip()
        data_infos.append(dict(filename=filename))
        skip_lines = int(lines[i + 2]) + 3
        i += skip_lines
    return data_infos


def get_image_metas(data_info, img_prefix):
    filename = data_info.get('filename', None)
    if filename is not None:
        if img_prefix is not None:
            filename = osp.join(img_prefix, filename)
        img_bytes = get(filename)
        img = mmcv.imfrombytes(img_bytes, flag='color')
        shape = img.shape
        meta = dict(filename=filename, ori_shape=shape)
    else:
        raise NotImplementedError('Missing `filename` in data_info')
    return meta


def main():
    args = parse_args()
    assert args.out.endswith('pkl'), 'The output file name must be pkl suffix'

    # load config files
    cfg = Config.fromfile(args.config)
    dataloader_cfg = cfg.get(f'{args.dataset}_dataloader')
    ann_file = osp.join(dataloader_cfg.dataset.data_root,
                        dataloader_cfg.dataset.ann_file)
    img_prefix = osp.join(dataloader_cfg.dataset.data_root,
                          dataloader_cfg.dataset.data_prefix['img'])

    print(f'{"-" * 5} Start Processing {"-" * 5}')
    if ann_file.endswith('csv'):
        data_infos = get_metas_from_csv_style_ann_file(ann_file)
    elif ann_file.endswith('txt'):
        data_infos = get_metas_from_txt_style_ann_file(ann_file)
    else:
        shuffix = ann_file.split('.')[-1]
        raise NotImplementedError('File name must be csv or txt suffix but '
                                  f'get {shuffix}')

    print(f'Successfully load annotation file from {ann_file}')
    print(f'Processing {len(data_infos)} images...')
    pool = Pool(args.nproc)
    # get image metas with multiple processes
    image_metas = pool.starmap(
        get_image_metas,
        zip(data_infos, [img_prefix for _ in range(len(data_infos))]),
    )
    pool.close()

    # save image metas
    root_path = dataloader_cfg.dataset.ann_file.rsplit('/', 1)[0]
    save_path = osp.join(root_path, args.out)
    dump(image_metas, save_path, protocol=4)
    print(f'Image meta file save to: {save_path}')


if __name__ == '__main__':
    main()
