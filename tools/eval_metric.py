import argparse

import mmcv
from mmcv import Config

from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl', help='Input result file in pickle format')
    parser.add_argument('json_prefix', help='Output json prefix')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl)
    dataset.format_results(outputs, jsonfile_prefix=args.json_prefix)
    if args.eval:
        dataset.evaluate(outputs, args.eval)


if __name__ == '__main__':
    main()
