import math
import torch
from torch.autograd import Function
from .. import masked_conv2d_cuda


class MaskedConv2dFunction(Function):

    @staticmethod
    def forward(ctx, features, mask, weight, bias, padding=0):
        assert mask.dim() == 3 and mask.size(0) == 1
        assert features.dim() == 4 and features.size(0) == 1
        assert features.size()[2:] == mask.size()[1:]
        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        elif isinstance(padding, tuple):
            assert len(padding) == 2
            assert isinstance(padding[0], int)
            assert isinstance(padding[1], int)
            pad_h, pad_w = padding
        else:
            raise TypeError(
                '"padding" must be an integer or tuple of integers')
        if not features.is_cuda:
            raise NotImplementedError

        stride_h = 1
        stride_w = 1

        out_channel, in_channel, kernel_h, kernel_w = weight.size()

        batch_size = features.size(0)
        out_h = int(
            math.floor((features.size(2) + 2 * pad_h -
                        (kernel_h - 1) - 1) / stride_h + 1))
        out_w = int(
            math.floor((features.size(3) + 2 * pad_w -
                        (kernel_h - 1) - 1) / stride_w + 1))
        mask_inds = torch.nonzero(mask[0] > 0)
        mask_h_idx = mask_inds[:, 0].contiguous()
        mask_w_idx = mask_inds[:, 1].contiguous()
        data_col = features.new(in_channel * kernel_h * kernel_w,
                                mask_inds.size(0)).zero_()
        masked_conv2d_cuda.masked_im2col_forward(
            features, mask_h_idx, mask_w_idx, kernel_h, kernel_w, pad_h, pad_w,
            data_col)

        masked_output = torch.addmm(1, bias[:, None], 1,
                                    weight.view(out_channel, -1), data_col)
        # masked_output = torch.mm(weight.view(out_channel, -1), data_col)
        output = features.new(batch_size, out_channel, out_h, out_w).zero_()
        masked_conv2d_cuda.masked_col2im_forward(masked_output, mask_h_idx,
                                                 mask_w_idx, out_h, out_w,
                                                 out_channel, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None, ) * 5


masked_conv2d = MaskedConv2dFunction.apply
