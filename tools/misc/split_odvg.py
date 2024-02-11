import argparse
import json
import os
import shutil

import jsonlines
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='The data root.')
    parser.add_argument('ann_file', type=str)
    parser.add_argument('img_prefix', type=str)
    parser.add_argument(
        'out_dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.')
    parser.add_argument(
        '--label-map-file', '-m', type=str, help='label map file')
    parser.add_argument(
        '--num-img',
        '-n',
        default=200,
        type=int,
        help='num of extract image, -1 means all images')
    parser.add_argument('--seed', default=-1, type=int, help='seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out_dir != args.data_root, \
        'The file will be overwritten in place, ' \
        'so the same folder is not allowed !'

    seed = int(args.seed)
    if seed != -1:
        print(f'Set the global seed: {seed}')
        np.random.seed(int(args.seed))

    ann_file = os.path.join(args.data_root, args.ann_file)
    with open(ann_file, 'r') as f:
        data_list = [json.loads(line) for line in f]

    np.random.shuffle(data_list)

    num_img = args.num_img

    progress_bar = ProgressBar(num_img)
    for i in range(num_img):
        file_name = data_list[i]['filename']
        image_path = os.path.join(args.data_root, args.img_prefix, file_name)
        out_image_dir = os.path.join(args.out_dir, args.img_prefix)
        mkdir_or_exist(out_image_dir)
        out_image_path = os.path.join(out_image_dir, file_name)
        shutil.copyfile(image_path, out_image_path)

        progress_bar.update()

    out_path = os.path.join(args.out_dir, args.ann_file)
    out_dir = os.path.dirname(out_path)
    mkdir_or_exist(out_dir)

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(data_list[:num_img])

    if args.label_map_file is not None:
        out_dir = os.path.dirname(
            os.path.join(args.out_dir, args.label_map_file))
        mkdir_or_exist(out_dir)
        shutil.copyfile(
            os.path.join(args.data_root, args.label_map_file),
            os.path.join(args.out_dir, args.label_map_file))


if __name__ == '__main__':
    main()
