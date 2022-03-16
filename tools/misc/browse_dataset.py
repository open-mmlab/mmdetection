# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [
            p for p in cfg.data.train.pipeline if p['type'] != 'SegRescale'
        ]
    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']
        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)

        gt_seg = item.get('gt_semantic_seg', None)
        if gt_seg is not None:
            pad_value = 255  # the padding value of gt_seg
            sem_labels = np.unique(gt_seg)
            all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
            all_labels, counts = np.unique(all_labels, return_counts=True)
            stuff_labels = all_labels[np.logical_and(counts < 2,
                                                     all_labels != pad_value)]
            stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
            gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
            gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                      axis=0)
            # If you need to show the bounding boxes,
            # please comment the following line
            gt_bboxes = None

        imshow_det_bboxes(
            item['img'],
            gt_bboxes,
            gt_labels,
            gt_masks,
            class_names=dataset.CLASSES,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=dataset.PALETTE,
            text_color=(200, 200, 200),
            mask_color=dataset.PALETTE)

        progress_bar.update()


if __name__ == '__main__':
    main()
