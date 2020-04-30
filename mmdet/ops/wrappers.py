"""
Modified from https://github.com/facebookresearch/detectron2/blob/master
/detectron2/layers/wrappers.py
Wrap some nn modules to support empty tensor input.
Currently, these wrappers are mainly used in mask heads like fcn_mask_head
and maskiou_heads since mask heads are trained on only positive RoIs.
"""
import math

import torch
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS
from torch.nn.modules.utils import _pair


class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


@CONV_LAYERS.register_module('Conv', force=True)
class Conv2d(nn.Conv2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4':
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4.0':
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super(ConvTranspose2d, self).forward(x)


class MaxPool2d(nn.MaxPool2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4':
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-2:], _pair(self.kernel_size),
                                     _pair(self.padding), _pair(self.stride),
                                     _pair(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)


class Linear(torch.nn.Linear):

    def forward(self, x):
        if x.numel() == 0:
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)
