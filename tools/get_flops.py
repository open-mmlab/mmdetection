import argparse
import json

import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.utils import ExtendedDictAction
from mmdet.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument('--out')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    if args.out:
        out = list()
        out.append({'key': 'size', 'display_name': 'Size', 'value': float(params.split(' ')[0]), 'unit': 'Mp'})
        out.append({'key': 'complexity', 'display_name': 'Complexity', 'value': 2 * float(flops.split(' ')[0]),
                    'unit': 'GFLOPs'})
        with open(args.out, 'w') as write_file:
            json.dump(out, write_file, indent=4)


if __name__ == '__main__':
    main()
