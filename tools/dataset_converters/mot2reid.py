# Copyright (c) OpenMMLab. All rights reserved.
# This script converts MOT dataset into ReID dataset.
# Official website of the MOT dataset: https://motchallenge.net/
#
# Label format of MOT dataset:
#   GTs:
#       <frame_id> # starts from 1,
#       <instance_id>, <x1>, <y1>, <w>, <h>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, <visibility>
#
#   DETs and Results:
#       <frame_id>, <instance_id>, <x1>, <y1>, <w>, <h>, <conf>,
#       <x>, <y>, <z> # for 3D objects
#
# Classes in MOT:
#   1: 'pedestrian'
#   2: 'person on vehicle'
#   3: 'car'
#   4: 'bicycle'
#   5: 'motorbike'
#   6: 'non motorized vehicle'
#   7: 'static person'
#   8: 'distractor'
#   9: 'occluder'
#   10: 'occluder on the ground',
#   11: 'occluder full'
#   12: 'reflection'
#
#   USELESS classes and IGNORES classes will not be selected
#   into the dataset for reid model training.
import argparse
import os
import os.path as osp
import random

import mmcv
import numpy as np
from mmengine.fileio import list_from_file
from tqdm import tqdm

USELESS = [3, 4, 5, 6, 9, 10, 11]
IGNORES = [2, 7, 8, 12, 13]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT dataset into ReID dataset.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument('-o', '--output', help='path to save ReID dataset')
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='proportion of the validation dataset to the whole ReID dataset')
    parser.add_argument(
        '--vis-threshold',
        type=float,
        default=0.3,
        help='threshold of visibility for each person')
    parser.add_argument(
        '--min-per-person',
        type=int,
        default=8,
        help='minimum number of images for each person')
    parser.add_argument(
        '--max-per-person',
        type=int,
        default=1000,
        help='maxmum number of images for each person')
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    in_folder = osp.join(args.input, 'train')
    video_names = os.listdir(in_folder)
    if 'MOT17' in in_folder:
        video_names = [
            video_name for video_name in video_names if 'FRCNN' in video_name
        ]
    is_mot15 = True if 'MOT15' in in_folder else False
    for video_name in tqdm(video_names):
        # load video infos
        video_folder = osp.join(in_folder, video_name)
        infos = list_from_file(f'{video_folder}/seqinfo.ini')
        # video-level infos
        assert video_name == infos[1].strip().split('=')[1]
        raw_img_folder = infos[2].strip().split('=')[1]
        raw_img_names = os.listdir(f'{video_folder}/{raw_img_folder}')
        raw_img_names = sorted(raw_img_names)
        num_raw_imgs = int(infos[4].strip().split('=')[1])
        assert num_raw_imgs == len(raw_img_names)

        reid_train_folder = osp.join(args.output, 'imgs')
        if not osp.exists(reid_train_folder):
            os.makedirs(reid_train_folder)
        gts = list_from_file(f'{video_folder}/gt/gt.txt')
        last_frame_id = -1
        for gt in gts:
            gt = gt.strip().split(',')
            frame_id, ins_id = map(int, gt[:2])
            ltwh = list(map(float, gt[2:6]))
            if is_mot15:
                class_id = 1
                visibility = 1.
            else:
                class_id = int(gt[7])
                visibility = float(gt[8])
            if class_id in USELESS:
                continue
            elif class_id in IGNORES:
                continue
            elif visibility < args.vis_threshold:
                continue
            reid_img_folder = osp.join(reid_train_folder,
                                       f'{video_name}_{ins_id:06d}')
            if not osp.exists(reid_img_folder):
                os.makedirs(reid_img_folder)
            idx = len(os.listdir(reid_img_folder))
            reid_img_name = f'{idx:06d}.jpg'
            if frame_id != last_frame_id:
                raw_img_name = raw_img_names[frame_id - 1]
                raw_img = mmcv.imread(
                    f'{video_folder}/{raw_img_folder}/{raw_img_name}')
                last_frame_id = frame_id
            xyxy = np.asarray(
                [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
            reid_img = mmcv.imcrop(raw_img, xyxy)
            mmcv.imwrite(reid_img, f'{reid_img_folder}/{reid_img_name}')

    reid_meta_folder = osp.join(args.output, 'meta')
    if not osp.exists(reid_meta_folder):
        os.makedirs(reid_meta_folder)
    reid_train_list = []
    reid_val_list = []
    reid_img_folder_names = sorted(os.listdir(reid_train_folder))
    num_ids = len(reid_img_folder_names)
    num_train_ids = int(num_ids * (1 - args.val_split))
    train_label, val_label = 0, 0
    random.seed(0)
    for reid_img_folder_name in reid_img_folder_names[:num_train_ids]:
        reid_img_names = os.listdir(
            f'{reid_train_folder}/{reid_img_folder_name}')
        # ignore ids whose number of image is less than min_per_person
        if (len(reid_img_names) < args.min_per_person):
            continue
        # downsampling when there are too many images owned by one id
        if (len(reid_img_names) > args.max_per_person):
            reid_img_names = random.sample(reid_img_names, args.max_per_person)
        # training set
        for reid_img_name in reid_img_names:
            reid_train_list.append(
                f'{reid_img_folder_name}/{reid_img_name} {train_label}\n')
        train_label += 1
    reid_entire_dataset_list = reid_train_list.copy()
    for reid_img_folder_name in reid_img_folder_names[num_train_ids:]:
        reid_img_names = os.listdir(
            f'{reid_train_folder}/{reid_img_folder_name}')
        # ignore ids whose number of image is less than min_per_person
        if (len(reid_img_names) < args.min_per_person):
            continue
        # downsampling when there are too many images owned by one id
        if (len(reid_img_names) > args.max_per_person):
            reid_img_names = random.sample(reid_img_names, args.max_per_person)
        for reid_img_name in reid_img_names:
            # validation set
            reid_val_list.append(
                f'{reid_img_folder_name}/{reid_img_name} {val_label}\n')
            reid_entire_dataset_list.append(
                f'{reid_img_folder_name}/{reid_img_name} '
                f'{train_label + val_label}\n')
        val_label += 1
    with open(
            osp.join(reid_meta_folder,
                     f'train_{int(100 * (1 - args.val_split))}.txt'),
            'w') as f:
        f.writelines(reid_train_list)
    with open(
            osp.join(reid_meta_folder, f'val_{int(100 * args.val_split)}.txt'),
            'w') as f:
        f.writelines(reid_val_list)
    with open(osp.join(reid_meta_folder, 'train.txt'), 'w') as f:
        f.writelines(reid_entire_dataset_list)


if __name__ == '__main__':
    main()
