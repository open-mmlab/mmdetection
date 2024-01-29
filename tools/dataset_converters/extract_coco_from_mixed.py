import argparse
import os.path as osp

import mmengine
from pycocotools.coco import COCO


def extract_coco(args):
    coco = COCO(args.mixed_ann)

    json_data = mmengine.load(args.mixed_ann)
    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }
    del json_data

    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        if img_info['data_source'] == 'coco':
            new_json_data['images'].append(img_info)
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            img_ann_info = coco.loadAnns(ann_ids)
            new_json_data['annotations'].extend(img_ann_info)
    if args.out_ann is None:
        out_ann = osp.dirname(
            args.mixed_ann) + '/final_mixed_train_only_coco.json'
        mmengine.dump(new_json_data, out_ann)
        print('save new json to {}'.format(out_ann))
    else:
        mmengine.dump(new_json_data, args.out_ann)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'split mixed goldg to coco.', add_help=True)
    parser.add_argument('mixed_ann', type=str)
    parser.add_argument('--out-ann', '-o', type=str)
    args = parser.parse_args()

    extract_coco(args)
