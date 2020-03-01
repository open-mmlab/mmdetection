import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import affine_grid_cuda


class _AffineGridGenerator(Function):

    @staticmethod
    def forward(ctx, theta, size, align_corners):

        ctx.save_for_backward(theta)
        ctx.size = size
        ctx.align_corners = align_corners

        func = affine_grid_cuda.affine_grid_generator_forward

        output = func(theta, size, align_corners)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        theta = ctx.saved_tensors
        size = ctx.size
        align_corners = ctx.align_corners

        func = affine_grid_cuda.affine_grid_generator_backward

        grad_input = func(grad_output, theta, size, align_corners)

        return grad_input, None, None


def affine_grid(theta, size, align_corners=False):
    if torch.__version__ >= '1.3':
        return F.affine_grid(theta, size, align_corners)
    elif align_corners:
        return F.affine_grid(theta, size)
    else:
        # enforce floating point dtype on theta
        if not theta.is_floating_point():
            raise ValueError(
                'Expected theta to have floating point type, but got {}'.
                format(theta.dtype))
        # check that shapes and sizes match
        if len(size) == 4:
            if theta.dim() != 3 or theta.size(-2) != 2 or theta.size(-1) != 3:
                raise ValueError(
                    'Expected a batch of 2D affine matrices of shape Nx2x3 '
                    'for size {}. Got {}.'.format(size, theta.shape))
        elif len(size) == 5:
            if theta.dim() != 3 or theta.size(-2) != 3 or theta.size(-1) != 4:
                raise ValueError(
                    'Expected a batch of 3D affine matrices of shape Nx3x4 '
                    'for size {}. Got {}.'.format(size, theta.shape))
        else:
            raise NotImplementedError(
                'affine_grid only supports 4D and 5D sizes, '
                'for 2D and 3D affine transforms, respectively. '
                'Got size {}.'.format(size))
        if min(size) <= 0:
            raise ValueError(
                'Expected non-zero, positive output size. Got {}'.format(size))
        return _AffineGridGenerator.apply(theta, size, align_corners)
