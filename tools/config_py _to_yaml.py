import os.path as osp
import sys
from importlib import import_module

import argparse
import yaml


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('input', help='Configuration file as *.py file')
    args.add_argument('output', help='Configuration file as *.yaml')

    return args.parse_args()


def main():
    args = parse_args()
    filename = osp.abspath(osp.expanduser(args.input))
    output = args.output
    assert output.endswith('.yaml') or output.endswith('.yml')

    assert osp.exists(filename)
    if filename.endswith('.py'):
        module_name = osp.basename(filename)[:-3]
        if '.' in module_name:
            raise ValueError('Dots are not allowed in config file path.')
        config_dir = osp.dirname(filename)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }

        with open(output, 'w') as f:
            yaml.dump(cfg_dict, f)


if __name__ == '__main__':
    main()
