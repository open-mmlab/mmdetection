from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class BaseMergeCell(nn.Module):

    def __init__(self,
                 channels=256,
                 binary_op_type='sum',
                 with_conv=True,
                 conv_cfg=dict(),
                 norm_cfg=None,
                 order=('act', 'conv', 'norm'),
                 with_input_conv_x=False,
                 with_input_conv_y=False,
                 input_conv_cfg=None,
                 input_norm_cfg=None,
                 resize_methods='interpolate'):
        super(BaseMergeCell, self).__init__()
        self.with_conv = with_conv
        self.with_x_conv = with_input_conv_x
        self.with_y_conv = with_input_conv_y
        self.resize_methods = resize_methods

        if self.with_conv:
            groups = conv_cfg.pop('groups', 1)
            kernel_size = conv_cfg.pop('kernel_size', 3)
            padding = conv_cfg.pop('kernel_size', 1)
            bias = conv_cfg.pop('bias', True)
            in_channels = channels \
                if binary_op_type == 'sum' \
                else channels * 2
            self.conv_out = ConvModule(
                in_channels,
                channels,
                kernel_size,
                groups=groups,
                bias=bias,
                padding=padding,
                norm_cfg=norm_cfg,
                order=order)

        self.op1 = self.build_input_conv(
            channels, input_conv_cfg,
            input_norm_cfg) if with_input_conv_x else nn.Sequential()
        self.op2 = self.build_input_conv(
            channels, input_conv_cfg,
            input_norm_cfg) if with_input_conv_y else nn.Sequential()

    def build_input_conv(self, channel, conv_cfg, norm_cfg):
        return ConvModule(
            channel,
            channel,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True)

    @abstractmethod
    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            if self.resize_methods == 'interpolate':
                return F.interpolate(x, size=size, mode='nearest')
            elif self.resize_methods == 'upsample':
                return nn.Upsample(
                    size=size, mode='bilinear', align_corners=False)(
                        x)
            else:
                raise NotImplementedError
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = x.shape[-1] // size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def forward(self, x1, x2, out_size=None):
        assert x1.shape[:2] == x2.shape[:2]
        assert out_size is None or len(out_size) == 2
        if out_size is None:  # resize to bigger one
            out_size = max(x1.size()[2:], x2.size()[2:])

        x1 = self.op1(x1)
        x2 = self.op2(x2)

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_conv:
            x = self.conv_out(x)
        return x


class SumCell(BaseMergeCell):

    def _binary_op(self, x1, x2):
        return x1 + x2


class ConcatCell(BaseMergeCell):

    def _binary_op(self, x1, x2):
        ret = torch.cat([x1, x2], dim=1)
        return ret


class GlobalPoolingCell(BaseMergeCell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1
