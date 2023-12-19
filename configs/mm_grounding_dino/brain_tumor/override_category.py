# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(description='Override Category')
    parser.add_argument('--data-root', default='data/brain_tumor_v2/')
    return parser.parse_args()


def main():
    args = parse_args()

    categories = [
        {
            "id": 0,
            "name": "brain-tumor",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "edema",
            "supercategory": "brain-tumor"
        },
        {
            "id": 2,
            "name": "non-enhancing tumor",
            "supercategory": "brain-tumor"
        },
        {
            "id": 3,
            "name": "enhancing tumor",
            "supercategory": "brain-tumor"
        }
    ]

    json_data = mmengine.load(args.data_root +
                              'valid/_annotations.coco.json')
    json_data['categories'] = categories
    mmengine.dump(json_data,
                  args.data_root + 'valid/_annotations_new_label.coco.json')

    json_data = mmengine.load(args.data_root +
                              'train/_annotations.coco.json')
    json_data['categories'] = categories
    mmengine.dump(json_data,
                  args.data_root + 'train/_annotations_new_label.coco.json')

    json_data = mmengine.load(args.data_root +
                              'test/_annotations.coco.json')
    json_data['categories'] = categories
    mmengine.dump(json_data,
                  args.data_root + 'test/_annotations_new_label.coco.json')


if __name__ == '__main__':
    main()
