# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make dummy results for MOT Challenge.')
    parser.add_argument('json_file', help='Input JSON file.')
    parser.add_argument('out_folder', help='Output folder.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    anns = mmengine.load(args.json_file)

    if not osp.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for video in anns['videos']:
        name = video['name']
        txt_name = f'{name}.txt'
        f = open(osp.join(args.out_folder, txt_name), 'wt')
        f.close()


if __name__ == '__main__':
    main()
