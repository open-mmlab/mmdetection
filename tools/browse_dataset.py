import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.datasets.builder import build_dataset


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
        type=int,
        default=999,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        mmcv.imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'] - 1,
            class_names=dataset.CLASSES,
            show=not args.not_show,
            out_file=filename,
            wait_time=args.show_interval)
        progress_bar.update()


if __name__ == '__main__':
    main()
