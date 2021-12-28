import torch
from torch import nn

from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS


class MemoryEfficientSwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


@ACTIVATION_LAYERS.register_module()
class MemoryEfficientSwish(nn.Module):
    def __init__(self, inplace):
        super(MemoryEfficientSwish, self).__init__()

    def forward(self, x):
        return MemoryEfficientSwishImplementation.apply(x)
