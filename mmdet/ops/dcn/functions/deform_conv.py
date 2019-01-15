import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .. import deform_conv_cuda


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                deformable_groups=1,
                im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new(*DeformConvFunction._output_size(
            input, weight, ctx.padding, ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new(), input.new()]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(input, torch.cuda.FloatTensor):
                    raise NotImplementedError

            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(
                input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(grad_output, torch.cuda.FloatTensor):
                    raise NotImplementedError

            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, ctx.bufs_[0], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.deformable_groups, cur_im2col_step)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.deformable_groups, 1, cur_im2col_step)

        return grad_input, grad_offset, grad_weight, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply
