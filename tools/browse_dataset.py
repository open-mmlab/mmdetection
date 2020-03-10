import argparse
import tempfile

import mmcv
from mmcv import Config
from tqdm import tqdm

from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--skip-type', nargs='+', default=['DefaultFormatBundle'])
    args = parser.parse_args()
    return args


def clear_config(config_path, skip_type):
    c = open(config_path).read()
    c = c.replace('to_rgb=True', 'to_rgb=False') \
        .replace('mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]', 'mean=[0] * 3, std=[1] * 3') \
        .replace("dict(type='Normalize', **img_norm_cfg),",
                 """dict(type='Normalize', **img_norm_cfg),
                    dict(type='ToUint8'),""")
    for s in skip_type:
        c = c.replace(s, 'Pass')
    print(c)
    ft = tempfile.NamedTemporaryFile('w+t', suffix='.py', delete=False)
    print(ft.name)
    ft.write(c)
    ft.close()

    cfg = Config.fromfile(ft.name)
    return cfg


def main():
    args = parse_args()

    cfg = clear_config(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    for item in tqdm(dataset):
        # print(item['img_meta'].data['filename'], item['gt_bboxes'], item['gt_labels'])
        mmcv.imshow_det_bboxes(item['img'], item['gt_bboxes'], item['gt_labels'], thickness=2)


if __name__ == '__main__':
    main()
