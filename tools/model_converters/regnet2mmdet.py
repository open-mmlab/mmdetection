# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch


def convert_stem(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('stem.conv', 'conv1')
    new_key = new_key.replace('stem.bn', 'bn1')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_head(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('head.fc', 'fc')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_reslayer(model_key, model_weight, state_dict, converted_names):
    split_keys = model_key.split('.')
    layer, block, module = split_keys[:3]
    block_id = int(block[1:])
    layer_name = f'layer{int(layer[1:])}'
    block_name = f'{block_id - 1}'

    if block_id == 1 and module == 'bn':
        new_key = f'{layer_name}.{block_name}.downsample.1.{split_keys[-1]}'
    elif block_id == 1 and module == 'proj':
        new_key = f'{layer_name}.{block_name}.downsample.0.{split_keys[-1]}'
    elif module == 'f':
        if split_keys[3] == 'a_bn':
            module_name = 'bn1'
        elif split_keys[3] == 'b_bn':
            module_name = 'bn2'
        elif split_keys[3] == 'c_bn':
            module_name = 'bn3'
        elif split_keys[3] == 'a':
            module_name = 'conv1'
        elif split_keys[3] == 'b':
            module_name = 'conv2'
        elif split_keys[3] == 'c':
            module_name = 'conv3'
        new_key = f'{layer_name}.{block_name}.{module_name}.{split_keys[-1]}'
    else:
        raise ValueError(f'Unsupported conversion of key {model_key}')
    print(f'Convert {model_key} to {new_key}')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert(src, dst):
    """Convert keys in pycls pretrained RegNet models to mmdet style."""
    # load caffe model
    regnet_model = torch.load(src)
    blobs = regnet_model['model_state']
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    for key, weight in blobs.items():
        if 'stem' in key:
            convert_stem(key, weight, state_dict, converted_names)
        elif 'head' in key:
            convert_head(key, weight, state_dict, converted_names)
        elif key.startswith('s'):
            convert_reslayer(key, weight, state_dict, converted_names)

    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print(f'not converted: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
