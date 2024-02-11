import argparse
import copy
import csv
import json
import os.path as osp

import jsonlines
from mmcv.image import imfrombytes
from mmengine.fileio import get


def _parse_label_file(label_file):
    index_list = []
    classes_names = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            classes_names.append(line[1])
            index_list.append(line[0])
    index_mapping = {index: i for i, index in enumerate(index_list)}
    return classes_names, index_mapping


# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None


def oi2odvg(args):
    ann_file = osp.join(args.input_dir, 'oidv6-train-annotations-bbox.csv')
    label_file = osp.join(args.input_dir, 'class-descriptions-boxable.csv')

    classes_names, index_mapping = _parse_label_file(label_file)

    label_map = {}
    for class_name, idx in index_mapping.items():
        class_name = classes_names[idx]
        label_map[str(idx)] = class_name

    if args.out_ann is None:
        output = osp.join(args.input_dir, 'openimages_label_map.json')
    else:
        output = osp.join(
            osp.dirname(args.out_ann), 'openimages_label_map.json')
    with open(output, 'w') as f:
        json.dump(label_map, f)

    metas = []
    skip_count = 0
    with open(ann_file, 'r') as f:
        reader = csv.reader(f)
        last_img_id = None
        _filename_shape = [0, 0]
        instances = []
        for i, line in enumerate(reader):
            if i == 0:
                continue
            img_id = line[0]
            if last_img_id is None:
                last_img_id = img_id
            label_id = line[2]

            filename = f'{img_id}.jpg'
            label = index_mapping[label_id]
            category = label_map[str(label)]
            bbox = [
                float(line[4]),  # xmin
                float(line[6]),  # ymin
                float(line[5]),  # xmax
                float(line[7])  # ymax
            ]

            # is_occluded = True if int(line[8]) == 1 else False
            # is_truncated = True if int(line[9]) == 1 else False
            is_group_of = True if int(line[10]) == 1 else False
            # is_depiction = True if int(line[11]) == 1 else False
            # is_inside = True if int(line[12]) == 1 else False

            # if any([is_occluded, is_truncated, is_group_of,
            # is_depiction, is_inside]):
            if is_group_of:
                print(f'skip {filename} of one instance')
                skip_count += 1
                continue

            # denormalize
            if filename != _filename_shape[0]:
                if args.img_prefix is not None:
                    _filename = osp.join(
                        osp.dirname(args.input_dir), args.img_prefix, filename)
                else:
                    _filename = osp.join(osp.dirname(args.input_dir), filename)
                img_bytes = get(_filename, backend_args)
                img = imfrombytes(img_bytes, flag='color')
                shape = img.shape
                _filename_shape = [filename, shape]
            else:
                shape = _filename_shape[1]

            h, w = shape[:2]
            bbox = [
                max(bbox[0] * w, 0),
                max(bbox[1] * h, 0),
                min(bbox[2] * w, w),
                min(bbox[3] * h, h)
            ]

            x1, y1, x2, y2 = bbox
            inter_w = max(0, min(x2, w) - max(x1, 0))
            inter_h = max(0, min(y2, h) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if w < 1 or h < 1:
                continue

            instance = {
                'filename': filename,
                'height': h,
                'width': w,
                'bbox': bbox,
                'label': label,
                'category': category
            }

            if img_id != last_img_id:
                copy_instances = copy.deepcopy(instances)
                for copy_instance in copy_instances:
                    _filename = copy_instance.pop('filename')
                    _h = copy_instance.pop('height')
                    _w = copy_instance.pop('width')

                meta_ifo = {
                    'filename': _filename,
                    'height': _h,
                    'width': _w,
                    'detection': {
                        'instances': copy_instances
                    }
                }
                metas.append(meta_ifo)
                instances = []
            instances.append(instance)
            last_img_id = img_id

        for instance in instances:
            _filename = instance.pop('filename')
            _h = instance.pop('height')
            _w = instance.pop('width')
        meta_ifo = {
            'filename': _filename,
            'height': _h,
            'width': _w,
            'detection': {
                'instances': instances
            }
        }
        metas.append(meta_ifo)

    if args.out_ann is None:
        out_path = osp.join(args.input_dir, 'oidv6-train-annotations_od.json')
    else:
        out_path = args.out_ann

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(metas)

    print('skip {} instances'.format(skip_count))
    print('save to {}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'openimages to odvg format.', add_help=True)
    parser.add_argument(
        '--input-dir',
        default='data/OpenImages/annotations',
        type=str,
        help='input list name')
    parser.add_argument('--img-prefix', default='OpenImages/train/')
    parser.add_argument('--out-ann', '-o', type=str)
    args = parser.parse_args()

    oi2odvg(args)
