from torch import nn
from torch.autograd import Function

from . import corner_pool_ext


class TopPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = corner_pool_ext.top_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = corner_pool_ext.top_pool_backward(input, grad_output)
        return output


class BottomPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = corner_pool_ext.bottom_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = corner_pool_ext.bottom_pool_backward(input, grad_output)
        return output


class LeftPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = corner_pool_ext.left_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = corner_pool_ext.left_pool_backward(input, grad_output)
        return output


class RightPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = corner_pool_ext.right_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = corner_pool_ext.right_pool_backward(input, grad_output)
        return output


class CornerPool(nn.Module):
    """Corner Pooling.

    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.

    Please refer to https://arxiv.org/abs/1808.01244 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.

    Args:
        mode(str): Pooling orientation for the pooling layer

            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling

    Returns:
        Feature map after pooling.
    """

    pool_functions = {
        'bottom': BottomPoolFunction,
        'left': LeftPoolFunction,
        'right': RightPoolFunction,
        'top': TopPoolFunction,
    }

    def __init__(self, mode):
        super(CornerPool, self).__init__()
        assert mode in self.pool_functions
        self.corner_pool = self.pool_functions[mode]

    def forward(self, x):
        return self.corner_pool.apply(x)
