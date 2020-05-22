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


class TopPool(nn.Module):

    def forward(self, x):
        return TopPoolFunction.apply(x)


class BottomPool(nn.Module):

    def forward(self, x):
        return BottomPoolFunction.apply(x)


class LeftPool(nn.Module):

    def forward(self, x):
        return LeftPoolFunction.apply(x)


class RightPool(nn.Module):

    def forward(self, x):
        return RightPoolFunction.apply(x)
