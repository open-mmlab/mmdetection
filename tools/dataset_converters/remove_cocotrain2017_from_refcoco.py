import argparse
import json
import os.path as osp

import mmengine
from pycocotools.coco import COCO


def diff_image_id(coco2017_train_ids, ref_ids):
    set1 = set(coco2017_train_ids)
    set2 = set(ref_ids)
    intersection = set1.intersection(set2)
    result = set1 - intersection
    return result


def gen_new_json(coco2017_train_path, json_data, coco2017_train_ids):
    coco = COCO(coco2017_train_path)
    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }

    for id in coco2017_train_ids:
        ann_ids = coco.getAnnIds(imgIds=[id])
        img_ann_info = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs([id])[0]

        new_json_data['images'].append(img_info)
        new_json_data['annotations'].extend(img_ann_info)
    return new_json_data


# coco2017 val and final_mixed_train.json have no intersection,
# so deduplication is not necessary.

# coco2017 val and datasets like refcoco based on coco2014 train
# have no intersection, so deduplication is not necessary.


# coco2017 train and datasets like refcoco based on coco2014
# train have overlapping annotations in the validation set,
# so deduplication is required.
def exclude_coco(args):
    with open(args.coco2017_train, 'r') as f:
        coco2017_train = json.load(f)
    coco2017_train_ids = [train['id'] for train in coco2017_train['images']]
    orig_len = len(coco2017_train_ids)

    with open(osp.join(args.mdetr_anno_dir, 'finetune_refcoco_val.json'),
              'r') as f:
        refcoco_ann = json.load(f)
    refcoco_ids = [refcoco['original_id'] for refcoco in refcoco_ann['images']]
    coco2017_train_ids = diff_image_id(coco2017_train_ids, refcoco_ids)

    with open(
            osp.join(args.mdetr_anno_dir, 'finetune_refcoco+_val.json'),
            'r') as f:
        refcoco_plus_ann = json.load(f)
    refcoco_plus_ids = [
        refcoco['original_id'] for refcoco in refcoco_plus_ann['images']
    ]
    coco2017_train_ids = diff_image_id(coco2017_train_ids, refcoco_plus_ids)

    with open(
            osp.join(args.mdetr_anno_dir, 'finetune_refcocog_val.json'),
            'r') as f:
        refcocog_ann = json.load(f)
    refcocog_ids = [
        refcoco['original_id'] for refcoco in refcocog_ann['images']
    ]
    coco2017_train_ids = diff_image_id(coco2017_train_ids, refcocog_ids)

    with open(
            osp.join(args.mdetr_anno_dir, 'finetune_grefcoco_val.json'),
            'r') as f:
        grefcoco_ann = json.load(f)
    grefcoco_ids = [
        refcoco['original_id'] for refcoco in grefcoco_ann['images']
    ]
    coco2017_train_ids = diff_image_id(coco2017_train_ids, grefcoco_ids)

    coco2017_train_ids = list(coco2017_train_ids)
    print(
        'remove {} images from coco2017_train'.format(orig_len -
                                                      len(coco2017_train_ids)))

    new_json_data = gen_new_json(args.coco2017_train, coco2017_train,
                                 coco2017_train_ids)
    if args.out_ann is None:
        out_ann = osp.dirname(
            args.coco2017_train) + '/instances_train2017_norefval.json'
        mmengine.dump(new_json_data, out_ann)
        print('save new json to {}'.format(out_ann))
    else:
        mmengine.dump(new_json_data, args.out_ann)
        print('save new json to {}'.format(args.out_ann))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('mdetr_anno_dir', type=str)
    parser.add_argument('coco2017_train', type=str)
    parser.add_argument('--out-ann', '-o', type=str)
    args = parser.parse_args()

    exclude_coco(args)
