import os.path as osp
import sys
from importlib import import_module

import argparse
import yaml
import json


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('input', help='Input configuration file as *.py file')
    args.add_argument('output', help='Output configuration file as *.yaml or *.json')

    return args.parse_args()


def main():
    args = parse_args()
    filename = osp.abspath(osp.expanduser(args.input))
    output = args.output

    assert output.endswith('.yaml') or output.endswith('.yml') or output.endswith('.json')
    assert osp.exists(filename)
    assert filename.endswith('.py')

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
        if output.endswith('.json'):
            json.dump(cfg_dict, f, indent=4)
        else:
            yaml.dump(cfg_dict, f)


if __name__ == '__main__':
    main()
