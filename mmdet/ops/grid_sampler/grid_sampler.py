import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import grid_sampler_ext


class _GridSampler(Function):

    @staticmethod
    def forward(ctx, input, grid, mode_enum, padding_mode_enum, align_corners):

        ctx.save_for_backward(input, grid)
        ctx.mode_enum = mode_enum
        ctx.padding_mode_enum = padding_mode_enum
        ctx.align_corners = align_corners

        output = grid_sampler_ext.grid_sampler_forward(input, grid, mode_enum,
                                                       padding_mode_enum,
                                                       align_corners)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        mode_enum = ctx.mode_enum
        padding_mode_enum = ctx.padding_mode_enum
        align_corners = ctx.align_corners

        grad_input, grad_grid = grid_sampler_ext.grid_sampler_backward(
            grad_output, input, grid, mode_enum, padding_mode_enum,
            align_corners)

        return grad_input, grad_grid, None, None, None


def grid_sample(input,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False):
    if torch.__version__ >= '1.3':
        return F.grid_sample(input, grid, mode, padding_mode, align_corners)
    elif align_corners:
        return F.grid_sample(input, grid, mode, padding_mode)
    else:

        # use self-compiled grid_sampler to support align_corners=False

        assert mode in ['bilinear', 'nearest'], \
            'expected mode to be bilinear or nearest, but got: {}'.format(mode)

        assert padding_mode in ['zeros', 'border', 'reflection'], \
            'expected padding_mode to be zeros, border, or reflection, ' \
            'but got: {}'.format(padding_mode)

        if mode == 'bilinear':
            mode_enum = 0
        else:
            mode_enum = 1

        if padding_mode == 'zeros':
            padding_mode_enum = 0
        elif padding_mode == 'border':
            padding_mode_enum = 1
        else:
            padding_mode_enum = 2

        # shape check
        assert input.device == grid.device, \
            'expected input and grid to be on same device, ' \
            'but input is on {} and grid is on {}'.format(
                input.device, grid.device)
        assert input.dtype == grid.dtype, \
            'expected input and grid to have the same dtype, ' \
            'but input has {} and grid has {}'.format(
                input.dtype, grid.dtype)
        assert input.dim() == 4 or input.dim() == 5, \
            'expected 4D or 5D input and grid with same number of dimensions' \
            'but got input with sizes {} and grid with sizes {}'.format(
                input.size(), grid.size())
        assert input.size(0) == grid.size(0), \
            'expected input and grid to have the same batch size, ' \
            'but got input with sizes {} and grid with sizes {}'.format(
                input.size(), grid.size())
        assert grid.size(-1) == input.dim() - 2, \
            'expected grid to have size {} in last {} dimension, ' \
            'but got grid with sizes '.format(
                input.dim() - 2, grid.size())
        for i in range(2, input.dim()):
            assert input.size(i) > 0, \
                'expected input to have non-empty spatial dimensions, ' \
                'but input has sizes {} with dimension {} being empty'.format(
                    input.sizes(), i)

        return _GridSampler.apply(input, grid, mode_enum, padding_mode_enum,
                                  align_corners)
