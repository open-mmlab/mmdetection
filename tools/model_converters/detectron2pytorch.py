# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import mmcv
import torch

arch_settings = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_bn(blobs, state_dict, caffe_name, torch_name, converted_names):
    # detectron replace bn with affine channel layer
    state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                              '_b'])
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_s'])
    bn_size = state_dict[torch_name + '.weight'].size()
    state_dict[torch_name + '.running_mean'] = torch.zeros(bn_size)
    state_dict[torch_name + '.running_var'] = torch.ones(bn_size)
    converted_names.add(caffe_name + '_b')
    converted_names.add(caffe_name + '_s')


def convert_conv_fc(blobs, state_dict, caffe_name, torch_name,
                    converted_names):
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_w'])
    converted_names.add(caffe_name + '_w')
    if caffe_name + '_b' in blobs:
        state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                                  '_b'])
        converted_names.add(caffe_name + '_b')


def convert(src, dst, depth):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # load arch_settings
    if depth not in arch_settings:
        raise ValueError('Only support ResNet-50 and ResNet-101 currently')
    block_nums = arch_settings[depth]
    # load caffe model
    caffe_model = mmcv.load(src, encoding='latin1')
    blobs = caffe_model['blobs'] if 'blobs' in caffe_model else caffe_model
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    convert_conv_fc(blobs, state_dict, 'conv1', 'conv1', converted_names)
    convert_bn(blobs, state_dict, 'res_conv1_bn', 'bn1', converted_names)
    for i in range(1, len(block_nums) + 1):
        for j in range(block_nums[i - 1]):
            if j == 0:
                convert_conv_fc(blobs, state_dict, f'res{i + 1}_{j}_branch1',
                                f'layer{i}.{j}.downsample.0', converted_names)
                convert_bn(blobs, state_dict, f'res{i + 1}_{j}_branch1_bn',
                           f'layer{i}.{j}.downsample.1', converted_names)
            for k, letter in enumerate(['a', 'b', 'c']):
                convert_conv_fc(blobs, state_dict,
                                f'res{i + 1}_{j}_branch2{letter}',
                                f'layer{i}.{j}.conv{k+1}', converted_names)
                convert_bn(blobs, state_dict,
                           f'res{i + 1}_{j}_branch2{letter}_bn',
                           f'layer{i}.{j}.bn{k + 1}', converted_names)
    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print(f'Not Convert: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('depth', type=int, help='ResNet model depth')
    args = parser.parse_args()
    convert(args.src, args.dst, args.depth)


if __name__ == '__main__':
    main()
