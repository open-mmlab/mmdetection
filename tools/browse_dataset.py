import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--skip-type',
                        type=str,
                        nargs='+',
                        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
                        help='skip some useless pipeline')
    parser.add_argument('--output-dir',
                        default=None,
                        type=str,
                        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--show-interval', type=float, default=1, help='the interval of show')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [x for x in train_data_cfg.pipeline if x['type'] not in skip_type]

    return cfg


def plot_annotation(args, filename, img, bboxes, labels, class_names=None):
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, (0, 255, 0), thickness=1)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    if args.output_dir is None:
        plt.show(block=False)
        plt.pause(1)
    else:
        mmcv.mkdir_or_exist(args.output_dir)
        out = os.path.join(args.output_dir, filename)
        plt.savefig(out)
        plt.cla()


def main():
    args = parse_args()
    if args.backend is not None:
        plt.switch_backend(args.backend)
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    class_name = {k+1: v for k, v in zip(range(len(dataset.CLASSES)), dataset.CLASSES)}

    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        filename = Path(item['filename']).name
        plot_annotation(args, filename, item['img'], item['gt_bboxes'], item['gt_labels'], class_name)
        progress_bar.update()


if __name__ == '__main__':
    main()
