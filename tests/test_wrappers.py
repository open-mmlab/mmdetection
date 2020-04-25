from collections import OrderedDict
from itertools import product
from unittest.mock import patch

import torch
import torch.nn as nn

from mmdet.ops import Conv2d, ConvTranspose2d, Linear, MaxPool2d

torch.__version__ = '1.1'  # force test


def test_conv2d():
    """
    CommandLine:
        xdoctest -m tests/test_wrappers.py test_conv2d
    """

    test_cases = OrderedDict([('in_w', [10, 20]), ('in_h', [10, 20]),
                              ('in_channel', [1, 3]), ('out_channel', [1, 3]),
                              ('kernel_size', [3, 5]), ('stride', [1, 2]),
                              ('padding', [0, 1]), ('dilation', [1, 2])])

    # train mode
    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x_empty = torch.randn(0, in_cha, in_h, in_w)
        torch.manual_seed(0)
        wrapper = Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x_empty)

        # torch op with 3-dim input as shape reference
        x_normal = torch.randn(3, in_cha, in_h, in_w).requires_grad_(True)
        torch.manual_seed(0)
        ref = nn.Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        ref_out = ref(x_normal)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_cha, in_h, in_w)
    wrapper = Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
    wrapper.eval()
    wrapper(x_empty)


def test_conv_transposed_2d():
    test_cases = OrderedDict([('in_w', [10, 20]), ('in_h', [10, 20]),
                              ('in_channel', [1, 3]), ('out_channel', [1, 3]),
                              ('kernel_size', [3, 5]), ('stride', [1, 2]),
                              ('padding', [0, 1]), ('dilation', [1, 2])])

    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x_empty = torch.randn(0, in_cha, in_h, in_w, requires_grad=True)
        # out padding must be smaller than either stride or dilation
        op = min(s, d) - 1
        torch.manual_seed(0)
        wrapper = ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        wrapper_out = wrapper(x_empty)

        # torch op with 3-dim input as shape reference
        x_normal = torch.randn(3, in_cha, in_h, in_w)
        torch.manual_seed(0)
        ref = nn.ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        ref_out = ref(x_normal)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_cha, in_h, in_w)
    wrapper = ConvTranspose2d(
        in_cha, out_cha, k, stride=s, padding=p, dilation=d, output_padding=op)
    wrapper.eval()
    wrapper(x_empty)


def test_max_pool_2d():
    test_cases = OrderedDict([('in_w', [10, 20]), ('in_h', [10, 20]),
                              ('in_channel', [1, 3]), ('out_channel', [1, 3]),
                              ('kernel_size', [3, 5]), ('stride', [1, 2]),
                              ('padding', [0, 1]), ('dilation', [1, 2])])

    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x_empty = torch.randn(0, in_cha, in_h, in_w, requires_grad=True)
        wrapper = MaxPool2d(k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x_empty)

        # torch op with 3-dim input as shape reference
        x_normal = torch.randn(3, in_cha, in_h, in_w)
        ref = nn.MaxPool2d(k, stride=s, padding=p, dilation=d)
        ref_out = ref(x_normal)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        assert torch.equal(wrapper(x_normal), ref_out)


def test_linear():
    test_cases = OrderedDict([
        ('in_w', [10, 20]),
        ('in_h', [10, 20]),
        ('in_feature', [1, 3]),
        ('out_feature', [1, 3]),
    ])

    for in_h, in_w, in_feature, out_feature in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x_empty = torch.randn(0, in_feature, requires_grad=True)
        torch.manual_seed(0)
        wrapper = Linear(in_feature, out_feature)
        wrapper_out = wrapper(x_empty)

        # torch op with 3-dim input as shape reference
        x_normal = torch.randn(3, in_feature)
        torch.manual_seed(0)
        ref = nn.Linear(in_feature, out_feature)
        ref_out = ref(x_normal)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_feature)
    wrapper = Linear(in_feature, out_feature)
    wrapper.eval()
    wrapper(x_empty)


def test_nn_op_forward_called():
    torch.__version__ = '1.4.1'

    for m in ['Conv2d', 'ConvTranspose2d', 'MaxPool2d']:
        with patch(f'torch.nn.{m}.forward') as nn_module_forward:
            # randn input
            x_empty = torch.randn(0, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_empty)
            nn_module_forward.assert_called_with(x_empty)

            # non-randn input
            x_normal = torch.randn(1, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_normal)
            nn_module_forward.assert_called_with(x_normal)

    with patch('torch.nn.Linear.forward') as nn_module_forward:
        # randn input
        x_empty = torch.randn(0, 3)
        wrapper = Linear(3, 3)
        wrapper(x_empty)
        nn_module_forward.assert_not_called()

        # non-randn input
        x_normal = torch.randn(1, 3)
        wrapper = Linear(3, 3)
        wrapper(x_normal)
        nn_module_forward.assert_called_with(x_normal)
