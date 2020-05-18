from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class BaseMergeCell(nn.Module):
    """The basic class for cells used in NAS-FPN and NAS-FCOS.

    BaseMergeCell takes 2 inputs. After applying concolution
    on them, they are resized to the target size. Then,
    they go through binary_op, which depends on the type of cell.

    """

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 with_conv=True,
                 conv_cfg=dict(),
                 norm_cfg=None,
                 order=('act', 'conv', 'norm'),
                 with_input1_conv=False,
                 with_input2_conv=False,
                 input_conv_cfg=None,
                 input_norm_cfg=None,
                 upsample_mode='nearest'):
        super(BaseMergeCell, self).__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.with_conv = with_conv
        self.with_input1_conv = with_input1_conv
        self.with_input2_conv = with_input2_conv
        self.upsample_mode = upsample_mode

        if self.with_conv:
            self.conv_out = ConvModule(
                in_channels,
                out_channels,
                kernel_size=conv_cfg['kernel_size'],
                groups=conv_cfg['groups'],
                bias=conv_cfg['bias'],
                padding=conv_cfg['padding'],
                norm_cfg=norm_cfg,
                order=order)

        self.conv1 = self._build_input_conv(
            out_channels, input_conv_cfg,
            input_norm_cfg) if with_input1_conv else nn.Sequential()
        self.conv2 = self._build_input_conv(
            out_channels, input_conv_cfg,
            input_norm_cfg) if with_input2_conv else nn.Sequential()

    def _build_input_conv(self, channel, conv_cfg, norm_cfg):
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
            return F.interpolate(x, size=size, mode=self.upsample_mode)
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

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

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
