from torch import nn
from torch.autograd import Function

from . import center_pool_ext


class TopPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = center_pool_ext.top_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = center_pool_ext.top_backward(input, grad_output)[0]
        return output


class BottomPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = center_pool_ext.bottom_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = center_pool_ext.bottom_backward(input, grad_output)[0]
        return output


class LeftPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = center_pool_ext.left_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = center_pool_ext.left_backward(input, grad_output)[0]
        return output


class RightPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = center_pool_ext.right_forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = center_pool_ext.right_backward(input, grad_output)[0]
        return output


class CenterPool(nn.Module):

    mode_functions = {
        'bottom': BottomPoolFunction,
        'left': LeftPoolFunction,
        'right': RightPoolFunction,
        'top': TopPoolFunction,
    }

    def __init__(self, mode):
        super(CenterPool, self).__init__()
        assert mode in self.mode_functions
        self.center_pool = self.mode_functions[mode]

    def forward(self, x):
        return self.center_pool.apply(x)
