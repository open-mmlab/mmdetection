import torch
from torch import nn as nn
from torch.nn import functional as F


__all__ = ['swish_jit', 'SwishJit', 'mish_jit', 'MishJit']
           #'hard_swish_jit', 'HardSwishJit', 'hard_sigmoid_jit', 'HardSigmoidJit']


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish_jit(x, inplace=False):
    # inplace ignored
    return SwishJitAutoFn.apply(x)


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish_jit(x, inplace=False):
    # inplace ignored
    return MishJitAutoFn.apply(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishJitAutoFn.apply(x)


# @torch.jit.script
# def hard_swish_jit(x, inplac: bool = False):
#     return x.mul(F.relu6(x + 3.).mul_(1./6.))
#
#
# class HardSwishJit(nn.Module):
#     def __init__(self, inplace: bool = False):
#         super(HardSwishJit, self).__init__()
#
#     def forward(self, x):
#         return hard_swish_jit(x)
#
#
# @torch.jit.script
# def hard_sigmoid_jit(x, inplace: bool = False):
#     return F.relu6(x + 3.).mul(1./6.)
#
#
# class HardSigmoidJit(nn.Module):
#     def __init__(self, inplace: bool = False):
#         super(HardSigmoidJit, self).__init__()
#
#     def forward(self, x):
#         return hard_sigmoid_jit(x)
