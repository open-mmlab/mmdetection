# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmengine
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS, VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--show', default=True, action='store_true')
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = mmengine.ProgressBar(len(dataset))
    for idx, item in enumerate(dataset):  # inputs data_samples
        data_sample = item['data_samples']
        input = item['inputs']
        for img_idx in range(len(data_sample)):
            img_data_sample = data_sample[img_idx]
            img_path = img_data_sample.img_path
            img = input[img_idx].permute(1, 2, 0).numpy()
            out_file = osp.join(
                args.output_dir,
                str(idx).zfill(6),
                f'img_{img_idx}.jpg') if args.output_dir is not None else None
            img = img[..., [2, 1, 0]]  # bgr to rgb
            visualizer.add_datasample(
                osp.basename(img_path),
                img,
                data_sample=img_data_sample,
                draw_pred=False,
                show=args.show,
                wait_time=args.show_interval,
                out_file=out_file)
            # Record file path mapping.
            if args.output_dir is not None:
                with open(
                        osp.join(args.output_dir,
                                 str(idx).zfill(6), 'info.txt'), 'a') as f:
                    f.write(f'The source filepath of img_{img_idx}.jpg'
                            f'is `{img_path}`.\n')
        progress_bar.update()


if __name__ == '__main__':
    main()
