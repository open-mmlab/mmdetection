# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmengine import Config, DictAction

from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--save-path',
        default=None,
        help='save path of whole config, suffixed with .py, .json or .yml')
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

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(f'Config:\n{cfg.pretty_text}')

    if args.save_path is not None:
        save_path = args.save_path

        suffix = os.path.splitext(save_path)[-1]
        assert suffix in ['.py', '.json', '.yml']

        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        cfg.dump(save_path)
        print(f'Config saving at {save_path}')


if __name__ == '__main__':
    main()
