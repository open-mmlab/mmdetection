import argparse

import torch
import torch.nn as nn
from mmcv.runner import save_checkpoint

from mmdet.apis import init_detector


def merge_conv_bn(conv, bn):
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    merged_conv_w = conv_w * factor.reshape([conv.out_channels, 1, 1, 1])
    merged_conv_b = (conv_b - bn.running_mean) * factor + bn.bias

    merged_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True)
    merged_conv.weight = nn.Parameter(merged_conv_w)
    merged_conv.bias = nn.Parameter(merged_conv_b)
    return merged_conv


def merge_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only merge BN that is after Conv
                continue
            merged_conv = merge_conv_bn(last_conv, child)
            m._modules[last_conv_name] = merged_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            merge_module(child)
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge Conv and BN layers in a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('out', help='output path of the converted model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint)
    # merge conv and bn layers of the model
    merged_model = merge_module(model)
    save_checkpoint(merged_model, args.out)


if __name__ == '__main__':
    main()
