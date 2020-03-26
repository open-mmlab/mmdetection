import torch
import torch.nn as nn


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
        if isinstance(child, nn.BatchNorm2d):
            merged_conv = merge_conv_bn(last_conv, child)
            m._modules[last_conv_name] = merged_conv
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            merge_module(child)
    return m
