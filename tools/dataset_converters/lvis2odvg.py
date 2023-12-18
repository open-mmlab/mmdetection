import argparse
import json
import os.path

import jsonlines
from lvis import LVIS
from tqdm import tqdm

key_list_lvis = [i for i in range(1203)]
val_list_lvis = [i for i in range(1, 1204)]


def dump_lvis_label_map(args):
    with open(args.input, 'r') as f:
        j = json.load(f)
    o_dict = {}
    for category in j['categories']:
        index = str(int(category['id']) - 1)
        name = category['name']
        o_dict[index] = name
    if args.output is None:
        output = os.path.dirname(args.input) + '/lvis_v1_label_map.json'
    else:
        output = os.path.dirname(args.output) + '/lvis_v1_label_map.json'
    with open(output, 'w') as f:
        json.dump(o_dict, f)


def lvis2odvg(args):
    lvis = LVIS(args.input)
    cats = lvis.load_cats(lvis.get_cat_ids())
    nms = {cat['id']: cat['name'] for cat in cats}
    metas = []
    if args.output is None:
        out_path = args.input[:-5] + '_od.json'
    else:
        out_path = args.output

    key_list = key_list_lvis
    val_list = val_list_lvis
    dump_lvis_label_map(args)

    for img_id, img_info in tqdm(lvis.imgs.items()):
        file_name = img_info['coco_url'].replace(
            'http://images.cocodataset.org/', '')
        ann_ids = lvis.get_ann_ids(img_ids=[img_id])
        raw_ann_info = lvis.load_anns(ann_ids)
        instance_list = []
        for ann in raw_ann_info:
            if ann.get('ignore', False):
                print(f'invalid ignore box of {ann}')
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                print(f'invalid wh box of {ann}')
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                print(f'invalid area box of {ann}, '
                      f'w={img_info["width"]}, h={img_info["height"]}')
                continue

            if ann.get('iscrowd', False):
                print(f'invalid iscrowd box of {ann}')
                continue

            bbox_xyxy = [x1, y1, x1 + w, y1 + h]
            label = ann['category_id']
            category = nms[label]
            ind = val_list.index(label)
            label_trans = key_list[ind]
            instance_list.append({
                'bbox': bbox_xyxy,
                'label': label_trans,
                'category': category
            })
        metas.append({
            'filename': file_name,
            'height': img_info['height'],
            'width': img_info['width'],
            'detection': {
                'instances': instance_list
            }
        })

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(metas)

    print('save to {}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lvis to odvg format.', add_help=True)
    parser.add_argument('input', type=str, help='input list name')
    parser.add_argument('--output', '-o', type=str, help='input list name')
    args = parser.parse_args()
    lvis2odvg(args)
