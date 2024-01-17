import argparse
import json
import os.path

base_classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'microwave', 'oven', 'toaster',
                'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

novel_classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 'elephant',
                 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife',
                 'cake', 'couch', 'keyboard', 'sink', 'scissors')


def filter_annotation(anno_dict, split_name_list, class_id_to_split):
    filtered_categories = []
    for item in anno_dict['categories']:
        if class_id_to_split.get(item['id']) in split_name_list:
            item['split'] = class_id_to_split.get(item['id'])
            filtered_categories.append(item)
    anno_dict['categories'] = filtered_categories

    filtered_images = []
    filtered_annotations = []
    useful_image_ids = set()
    for item in anno_dict['annotations']:
        if class_id_to_split.get(item['category_id']) in split_name_list:
            filtered_annotations.append(item)
            useful_image_ids.add(item['image_id'])
    for item in anno_dict['images']:
        if item['id'] in useful_image_ids:
            filtered_images.append(item)
    anno_dict['annotations'] = filtered_annotations
    anno_dict['images'] = filtered_images


def coco2ovd(args):
    ann_path = os.path.join(args.data_root, 'annotations/')
    with open(ann_path + 'instances_train2017.json', 'r') as fin:
        coco_train_anno_all = json.load(fin)

    class_id_to_split = {}
    for item in coco_train_anno_all['categories']:
        if item['name'] in base_classes:
            class_id_to_split[item['id']] = 'seen'
        elif item['name'] in novel_classes:
            class_id_to_split[item['id']] = 'unseen'

    filter_annotation(coco_train_anno_all, ['seen'], class_id_to_split)
    with open(ann_path + 'instances_train2017_seen_2.json', 'w') as fout:
        json.dump(coco_train_anno_all, fout)

    with open(ann_path + 'instances_val2017.json', 'r') as fin:
        coco_val_anno_all = json.load(fin)

    filter_annotation(coco_val_anno_all, ['seen', 'unseen'], class_id_to_split)
    with open(ann_path + 'instances_val2017_all_2.json', 'w') as fout:
        json.dump(coco_val_anno_all, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to ovd format.', add_help=True)
    parser.add_argument('data_root', type=str, help='coco root path')
    args = parser.parse_args()

    coco2ovd(args)
