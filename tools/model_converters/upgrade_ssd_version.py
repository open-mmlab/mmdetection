import argparse
import tempfile
from collections import OrderedDict

import torch
from mmcv import Config


def parse_config(config_strings):
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)

    config = Config.fromfile(config_path)
    # check whether it is SSD
    if config.model.bbox_head.type != 'SSDHead':
        raise AssertionError('This is not a SSD model.')


def convert(in_file, out_file):
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    meta_info = checkpoint['meta']
    parse_config('#' + meta_info['config'])
    for key, value in in_state_dict.items():
        if 'extra' in key:
            layer_idx = int(key.split('.')[2])
            new_key = 'neck.extra_layers.{}.{}.conv.'.format(
                layer_idx // 2, layer_idx % 2) + key.split('.')[-1]
        elif 'l2_norm' in key:
            new_key = 'neck.l2_norm.weight'
        elif 'bbox_head' in key:
            new_key = key[:21] + '.0' + key[21:]
        else:
            new_key = key
        out_state_dict[new_key] = value
    checkpoint['state_dict'] = out_state_dict

    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser(description='Upgrade SSD version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')

    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
