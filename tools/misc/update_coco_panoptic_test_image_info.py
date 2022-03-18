import argparse
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update COCO test image information')
    parser.add_argument('data_root', help='Path to COCO annotation directory.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_root = args.data_root
    val_info = mmcv.load(osp.join(data_root, 'panoptic_val2017.json'))
    test_old_info = mmcv.load(
        osp.join(data_root, 'image_info_test-dev2017.json'))

    test_info = test_old_info
    test_info.update({'categories': val_info['categories']})
    mmcv.dump(test_info,
              osp.join(data_root, 'panoptic_image_info_test-dev2017.json'))


if __name__ == '__main__':
    main()
