# Copyright (c) OpenMMLab. All rights reserved.
# This script converts MOT labels into COCO style.
# Official website of the MOT dataset: https://motchallenge.net/
#
# Label format of MOT dataset:
#   GTs:
#       <frame_id> # starts from 1 but COCO style starts from 0,
#       <instance_id>, <x1>, <y1>, <w>, <h>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, <visibility>
#
#   DETs and Results:
#       <frame_id>, <instance_id>, <x1>, <y1>, <w>, <h>, <conf>,
#       <x>, <y>, <z> # for 3D objects

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import numpy as np
from tqdm import tqdm

# Classes in MOT:
CLASSES = [
    dict(id=1, name='pedestrian'),
    dict(id=2, name='person_on_vehicle'),
    dict(id=3, name='car'),
    dict(id=4, name='bicycle'),
    dict(id=5, name='motorbike'),
    dict(id=6, name='non_mot_vehicle'),
    dict(id=7, name='static_person'),
    dict(id=8, name='distractor'),
    dict(id=9, name='occluder'),
    dict(id=10, name='occluder_on_ground'),
    dict(id=11, name='occluder_full'),
    dict(id=12, name='reflection'),
    dict(id=13, name='crowd')
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT label and detections to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    parser.add_argument(
        '--convert-det',
        action='store_true',
        help='convert official detection results.')
    parser.add_argument(
        '--split-train',
        action='store_true',
        help='split the train set into half-train and half-validate.')
    return parser.parse_args()


def parse_gts(gts, is_mot15):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        if is_mot15:
            conf = 1.
            category_id = 1
            visibility = 1.
        else:
            conf = float(gt[6])
            category_id = int(gt[7])
            visibility = float(gt[8])
        anns = dict(
            category_id=category_id,
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf)
        outputs[frame_id].append(anns)
    return outputs


def parse_dets(dets):
    outputs = defaultdict(list)
    for det in dets:
        det = det.strip().split(',')
        frame_id, ins_id = map(int, det[:2])
        assert ins_id == -1
        bbox = list(map(float, det[2:7]))
        # [x1, y1, x2, y2] to be consistent with mmdet
        bbox = [
            bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[4]
        ]
        outputs[frame_id].append(bbox)

    return outputs


def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)

    sets = ['train', 'test']
    if args.split_train:
        sets += ['half-train', 'half-val']
    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        if 'half' in subset:
            in_folder = osp.join(args.input, 'train')
        else:
            in_folder = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')
        outputs = defaultdict(list)
        outputs['categories'] = CLASSES
        if args.convert_det:
            det_file = osp.join(args.output, f'{subset}_detections.pkl')
            detections = dict(det_bboxes=dict())
        video_names = os.listdir(in_folder)
        for video_name in tqdm(video_names):
            # basic params
            parse_gt = 'test' not in subset
            ins_maps = dict()
            # load video infos
            video_folder = osp.join(in_folder, video_name)
            infos = mmengine.list_from_file(f'{video_folder}/seqinfo.ini')
            # video-level infos
            assert video_name == infos[1].strip().split('=')[1]
            img_folder = infos[2].strip().split('=')[1]
            img_names = os.listdir(f'{video_folder}/{img_folder}')
            img_names = sorted(img_names)
            fps = int(infos[3].strip().split('=')[1])
            num_imgs = int(infos[4].strip().split('=')[1])
            assert num_imgs == len(img_names)
            width = int(infos[5].strip().split('=')[1])
            height = int(infos[6].strip().split('=')[1])
            video = dict(
                id=vid_id,
                name=video_name,
                fps=fps,
                width=width,
                height=height)
            # parse annotations
            if parse_gt:
                gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
                if 'MOT15' in video_folder:
                    img2gts = parse_gts(gts, True)
                else:
                    img2gts = parse_gts(gts, False)
            if args.convert_det:
                dets = mmengine.list_from_file(f'{video_folder}/det/det.txt')
                img2dets = parse_dets(dets)
            # make half sets
            if 'half' in subset:
                split_frame = num_imgs // 2 + 1
                if 'train' in subset:
                    img_names = img_names[:split_frame]
                elif 'val' in subset:
                    img_names = img_names[split_frame:]
                else:
                    raise ValueError(
                        'subset must be named with `train` or `val`')
                mot_frame_ids = [str(int(_.split('.')[0])) for _ in img_names]
                with open(f'{video_folder}/gt/gt_{subset}.txt', 'wt') as f:
                    for gt in gts:
                        if gt.split(',')[0] in mot_frame_ids:
                            f.writelines(f'{gt}\n')
            # image and box level infos
            for frame_id, name in enumerate(img_names):
                img_name = osp.join(video_name, img_folder, name)
                mot_frame_id = int(name.split('.')[0])
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id)
                if parse_gt:
                    gts = img2gts[mot_frame_id]
                    for gt in gts:
                        gt.update(id=ann_id, image_id=img_id)
                        mot_ins_id = gt['mot_instance_id']
                        if mot_ins_id in ins_maps:
                            gt['instance_id'] = ins_maps[mot_ins_id]
                        else:
                            gt['instance_id'] = ins_id
                            ins_maps[mot_ins_id] = ins_id
                            ins_id += 1
                        outputs['annotations'].append(gt)
                        ann_id += 1
                if args.convert_det:
                    dets = np.array(img2dets[mot_frame_id])
                    if dets.ndim == 1:
                        assert len(dets) == 0
                        dets = np.zeros((0, 5))
                    detections['det_bboxes'][img_name] = [dets]
                outputs['images'].append(image)
                img_id += 1
            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id
        print(f'{subset} has {ins_id} instances.')
        mmengine.dump(outputs, out_file)
        if args.convert_det:
            mmengine.dump(detections, det_file)
            print(f'Done! Saved as {out_file} and {det_file}')
        else:
            print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
