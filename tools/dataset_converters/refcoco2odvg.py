"""
```text
data
├── coco
│   ├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
│   ├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
|   ├── refcocog
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(umd).p
|   |── train2014
```
"""

import argparse
import os.path as osp
import mmengine
import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='refcoco to odvg')
    parser.add_argument('data_root')
    # refcoco refcoco+ refcocog
    # The annotation json of the original coco2014.
    parser.add_argument('--ann-file', default='refcocog/instances.json')
    # refcoco/refs(unc).p refcoco+/refs(unc).p refcocog/refs(umd).p
    parser.add_argument('--split-file', default='refcocog/refs(umd).p')
    parser.add_argument('--split', default='train')  # train/val/testA/testB
    args = parser.parse_args()
    return args


def init_refs(instances, splits):
    anns, imgs = {}, {}
    for ann in instances['annotations']:
        anns[ann['id']] = ann
    for img in instances['images']:
        imgs[img['id']] = img

    refs, ref_to_ann, ref_to_img = {}, {}, {}
    for ref in splits:
        # ids
        ref_id = ref['ref_id']
        ann_id = ref['ann_id']
        img_id = ref['image_id']
        # add mapping related to ref
        refs[ref_id] = ref
        ref_to_ann[ref_id] = anns[ann_id]
        ref_to_img[ref_id] = imgs[img_id]
    return refs, ref_to_ann, ref_to_img


def main():
    args = parse_args()
    split_file = osp.join(args.data_root, args.split_file)
    ann_file = osp.join(args.data_root, args.ann_file)

    splits = mmengine.load(split_file, file_format='pkl')
    instances = mmengine.load(ann_file, file_format='json')
    refs, ref_to_ann, ref_to_img = init_refs(instances, splits)

    ref_ids = [
        ref['ref_id'] for ref in splits if ref['split'] == args.split
    ]

    metas = []
    out_path = ann_file[:-5] + '_rec.json'

    for id in tqdm(ref_ids):
        out_dict = {}

        ref = refs[id]
        ann = ref_to_ann[id]
        img_info = ref_to_img[id]

        if ann.get('iscrowd', 0) == 1:
            continue

        out_dict['filename'] = img_info['file_name']
        out_dict['height'] = img_info['height']
        out_dict['width'] = img_info['width']
        bbox = ann['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

        sentence = [s['sent'] for s in ref['sentences']]
        instance = {'bbox': bbox, 'phrase': list(set(sentence))}
        out_dict['referring'] = [instance]

        metas.append(out_dict)

    print('  == dump meta ...')
    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(metas)
    print(f'save to {out_path}')


if __name__ == '__main__':
    main()
