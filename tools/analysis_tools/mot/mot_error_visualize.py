# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re

import mmcv
import motmetrics as mm
import numpy as np
import pandas as pd
from mmengine import Config
from mmengine.logging import print_log
from mmengine.registry import init_default_scope
from torch.utils.data import Dataset

from mmdet.registry import DATASETS
from mmdet.utils import imshow_mot_errors


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize errors for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument(
        '--result-dir', help='directory of the inference result')
    parser.add_argument(
        '--output-dir',
        help='directory where painted images or videos will be saved')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    parser.add_argument(
        '--fps', type=int, default=3, help='FPS of the output video')
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cv2', 'plt'],
        default='cv2',
        help='backend of visualization')
    args = parser.parse_args()
    return args


def compare_res_gts(results_dir: str, dataset: Dataset, video_name: str):
    """Evaluate the results of the video.

    Args:
        results_dir (str): the directory of the MOT results.
        dataset (Dataset): MOT dataset of the video to be evaluated.
        video_name (str): Name of the video to be evaluated.

    Returns:
        tuple: (acc, res, gt), acc contains the results of MOT metrics,
        res is the results of inference and gt is the ground truth.
    """
    if 'half-train' in dataset.ann_file:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt_half-train.txt')
        gt = mm.io.loadtxt(gt_file)
        gt.index = gt.index.set_levels(
            pd.factorize(gt.index.levels[0])[0] + 1, level=0)
    elif 'half-val' in dataset.ann_file:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt_half-val.txt')
        gt = mm.io.loadtxt(gt_file)
        gt.index = gt.index.set_levels(
            pd.factorize(gt.index.levels[0])[0] + 1, level=0)
    else:
        gt_file = osp.join(dataset.data_prefix['img_path'],
                           f'{video_name}/gt/gt.txt')
        gt = mm.io.loadtxt(gt_file)
        gt.index = gt.index.set_levels(
            pd.factorize(gt.index.levels[0])[0] + 1, level=0)
    res_file = osp.join(results_dir, f'{video_name}.txt')
    res = mm.io.loadtxt(res_file)
    ini_file = osp.join(dataset.data_prefix['img_path'],
                        f'{video_name}/seqinfo.ini')
    if osp.exists(ini_file):
        acc, _ = mm.utils.CLEAR_MOT_M(gt, res, ini_file)
    else:
        acc = mm.utils.compare_to_groundtruth(gt, res)

    return acc, res, gt


def main():
    args = parse_args()

    assert args.show or args.out_dir, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--show" or "--out-dir"')

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    print_log('This script visualizes the error for multiple object tracking. '
              'By Default, the red bounding box denotes false positive, '
              'the yellow bounding box denotes the false negative '
              'and the blue bounding box denotes ID switch.')

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # create index from frame_id to filename
    filenames_dict = dict()
    for i in range(len(dataset)):
        video_info = dataset.get_data_info(i)
        # the `data_info['file_name']` usually has the same format
        # with "MOT17-09-DPM/img1/000003.jpg"
        # split with both '\' and '/' to be compatible with different OS.
        for data_info in video_info['images']:
            split_path = re.split(r'[\\/]', data_info['file_name'])
            video_name = split_path[-3]
            frame_id = int(data_info['frame_id'] + 1)
            if video_name not in filenames_dict:
                filenames_dict[video_name] = dict()
        # the data_info['img_path'] usually has the same format
        # with `img_path_prefix + "MOT17-09-DPM/img1/000003.jpg"`
            filenames_dict[video_name][frame_id] = data_info['img_path']
    video_names = tuple(filenames_dict.keys())

    for video_name in video_names:
        print_log(f'Start processing video {video_name}')

        acc, res, gt = compare_res_gts(args.result_dir, dataset, video_name)

        frames_id_list = sorted(
            list(set(acc.mot_events.index.get_level_values(0))))
        for frame_id in frames_id_list:
            # events in the current frame
            events = acc.mot_events.xs(frame_id)
            cur_res = res.loc[frame_id] if frame_id in res.index else None
            cur_gt = gt.loc[frame_id] if frame_id in gt.index else None
            # path of image
            img = filenames_dict[video_name][frame_id]
            fps = events[events.Type == 'FP']
            fns = events[events.Type == 'MISS']
            idsws = events[events.Type == 'SWITCH']

            bboxes, ids, error_types = [], [], []
            for fp_index in fps.index:
                hid = events.loc[fp_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # error_type = 0 denotes false positive error
                error_types.append(0)
            for fn_index in fns.index:
                oid = events.loc[fn_index].OId
                bboxes.append([
                    cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
                    cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
                    cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
                    cur_gt.loc[oid].Confidence
                ])
                ids.append(-1)
                # error_type = 1 denotes false negative error
                error_types.append(1)
            for idsw_index in idsws.index:
                hid = events.loc[idsw_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # error_type = 2 denotes id switch
                error_types.append(2)
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                bboxes = np.asarray(bboxes, dtype=np.float32)
            ids = np.asarray(ids, dtype=np.int32)
            error_types = np.asarray(error_types, dtype=np.int32)
            imshow_mot_errors(
                img,
                bboxes,
                ids,
                error_types,
                show=args.show,
                out_file=osp.join(args.out_dir,
                                  f'{video_name}/{frame_id:06d}.jpg')
                if args.out_dir else None,
                backend=args.backend)

        print_log(f'Done! Visualization images are saved in '
                  f'\'{args.out_dir}/{video_name}\'')

        mmcv.frames2video(
            f'{args.out_dir}/{video_name}',
            f'{args.out_dir}/{video_name}.mp4',
            fps=args.fps,
            fourcc='mp4v',
            start=frames_id_list[0],
            end=frames_id_list[-1],
            show_progress=False)
        print_log(
            f'Done! Visualization video is saved as '
            f'\'{args.out_dir}/{video_name}.mp4\' with a FPS of {args.fps}')


if __name__ == '__main__':
    main()
