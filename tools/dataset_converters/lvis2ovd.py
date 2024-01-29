import argparse
import json
import os.path

import jsonlines


def lvis2ovd(args):
    ann_path = os.path.join(args.data_root, 'annotations/')

    lvis = json.load(open(ann_path + 'lvis_v1_val.json'))
    base_class_ids = [
        cat['id'] - 1 for cat in lvis['categories'] if cat['frequency'] != 'r'
    ]

    with open(ann_path + 'lvis_v1_train_od.json') as f:
        data = [json.loads(d) for d in f]
    for i in range(len(data)):
        instance = [
            inst for inst in data[i]['detection']['instances']
            if inst['label'] in base_class_ids
        ]
        data[i]['detection']['instances'] = instance
    with jsonlines.open(
            ann_path + 'lvis_v1_train_od_norare.json', mode='w') as writer:
        writer.write_all(data)

    label_map = json.load(open(ann_path + 'lvis_v1_label_map.json'))
    label_map = {
        k: v
        for k, v in label_map.items() if int(k) in base_class_ids
    }
    json.dump(label_map, open(ann_path + 'lvis_v1_label_map_norare.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lvis to ovd format.', add_help=True)
    parser.add_argument('data_root', type=str, help='coco root path')
    args = parser.parse_args()

    lvis2ovd(args)
