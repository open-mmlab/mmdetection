import torch
from torch.autograd import Function

from .. import carafe_naive_cuda


class CARAFENAIVEFunction(Function):

    @staticmethod
    def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        if features.is_cuda:
            carafe_naive_cuda.forward(features, masks, kernel_size, group_size,
                                      scale_factor, output)
        else:
            raise NotImplementedError

        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        features, masks = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        grad_input = torch.zeros_like(features)
        grad_masks = torch.zeros_like(masks)
        carafe_naive_cuda.backward(grad_output.contiguous(), features, masks,
                                   kernel_size, group_size, scale_factor,
                                   grad_input, grad_masks)

        return grad_input, grad_masks, None, None, None


carafe_naive = CARAFENAIVEFunction.apply
