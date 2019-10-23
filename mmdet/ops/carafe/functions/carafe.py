import torch
from torch.autograd import Function

from .. import carafe_cuda, carafe_cuda_benchmark


class CARAFEFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                masks,
                kernel_size,
                group_size,
                scale_factor,
                benchmark=False):
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
        ctx.benchmark = benchmark

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        routput = features.new_zeros(output.size(), requires_grad=False)
        rfeatures = features.new_zeros(features.size(), requires_grad=False)
        rmasks = masks.new_zeros(masks.size(), requires_grad=False)
        if features.is_cuda:
            if not benchmark:
                carafe_cuda.forward(features, rfeatures, masks, rmasks,
                                    kernel_size, group_size, scale_factor,
                                    routput, output)
            else:
                carafe_cuda_benchmark.forward(features, rfeatures, masks,
                                              rmasks, kernel_size, group_size,
                                              scale_factor, routput, output)
        else:
            raise NotImplementedError

        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks, rfeatures)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        features, masks, rfeatures = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor
        benchmark = ctx.benchmark

        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_masks = torch.zeros_like(masks, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_masks = torch.zeros_like(masks, requires_grad=False)
        if not benchmark:
            carafe_cuda.backward(grad_output.contiguous(), rfeatures, masks,
                                 kernel_size, group_size, scale_factor,
                                 rgrad_output, rgrad_input_hs, rgrad_input,
                                 rgrad_masks, grad_input, grad_masks)
        else:
            carafe_cuda_benchmark.backward(grad_output.contiguous(), rfeatures,
                                           masks, kernel_size, group_size,
                                           scale_factor, rgrad_output,
                                           rgrad_input_hs, rgrad_input,
                                           rgrad_masks, grad_input, grad_masks)
        return grad_input, grad_masks, None, None, None, None


carafe = CARAFEFunction.apply
