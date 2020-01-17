import math
import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from .. import masked_conv2d_cuda


class MaskedConv2dFunction(Function):

    @staticmethod
    def forward(ctx, features, mask, weight, bias, padding=0, stride=1):
        assert mask.dim() == 3 and mask.size(0) == 1
        assert features.dim() == 4 and features.size(0) == 1
        assert features.size()[2:] == mask.size()[1:]
        pad_h, pad_w = _pair(padding)
        stride_h, stride_w = _pair(stride)
        if stride_h != 1 or stride_w != 1:
            raise ValueError(
                'Stride could not only be 1 in masked_conv2d currently.')
        if not features.is_cuda:
            raise NotImplementedError

        out_channel, in_channel, kernel_h, kernel_w = weight.size()

        batch_size = features.size(0)
        out_h = int(
            math.floor((features.size(2) + 2 * pad_h -
                        (kernel_h - 1) - 1) / stride_h + 1))
        out_w = int(
            math.floor((features.size(3) + 2 * pad_w -
                        (kernel_h - 1) - 1) / stride_w + 1))
        mask_inds = torch.nonzero(mask[0] > 0)
        output = features.new_zeros(batch_size, out_channel, out_h, out_w)
        if mask_inds.numel() > 0:
            mask_h_idx = mask_inds[:, 0].contiguous()
            mask_w_idx = mask_inds[:, 1].contiguous()
            data_col = features.new_zeros(in_channel * kernel_h * kernel_w,
                                          mask_inds.size(0))
            masked_conv2d_cuda.masked_im2col_forward(features, mask_h_idx,
                                                     mask_w_idx, kernel_h,
                                                     kernel_w, pad_h, pad_w,
                                                     data_col)

            masked_output = torch.addmm(1, bias[:, None], 1,
                                        weight.view(out_channel, -1), data_col)
            masked_conv2d_cuda.masked_col2im_forward(masked_output, mask_h_idx,
                                                     mask_w_idx, out_h, out_w,
                                                     out_channel, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None, ) * 5


masked_conv2d = MaskedConv2dFunction.apply
