import argparse
import json
import os
from os import path as osp
import sys


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', help='path to the label dir')
    parser.add_argument('det_path', help='path to output detection file')
    args = parser.parse_args()

    return args


def label2det(label):
    boxes = list()
    for frame in label['frames']:
        for obj in frame['objects']:
            if 'box2d' not in obj:
                continue
            xy = obj['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            box = {'name': label['name'],
                   'timestamp': frame['timestamp'],
                   'category': obj['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            boxes.append(box)
    return boxes


def change_dir(label_dir, det_path):
    if not osp.exists(label_dir):
        print('Can not find', label_dir)
        return
    print('Processing', label_dir)
    input_names = [n for n in os.listdir(label_dir)
                   if osp.splitext(n)[1] == '.json']
    boxes = []
    count = 0
    for name in input_names:
        in_path = osp.join(label_dir, name)
        out = label2det(json.load(open(in_path, 'r')))
        boxes.extend(out)
        count += 1
        if count % 1000 == 0:
            print('Finished', count)
    with open(det_path, 'w') as fp:
        json.dump(boxes, fp, indent=4, separators=(',', ': '))


def main():
    args = parse_args()
    change_dir(args.label_dir, args.det_path)


if __name__ == '__main__':
    main()
