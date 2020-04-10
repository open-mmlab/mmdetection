from itertools import product
from unittest.mock import patch

import torch
import torch.nn as nn

from mmdet.ops import Conv2d, ConvTranspose2d, Linear, MaxPool2d

torch.__version__ = '1.1'  # force test


def test_conv_2d():
    test_cases = {
        'in_w': [10, 20],
        'in_h': [10, 20],
        'in_channel': [1, 3],
        'out_channel': [1, 3],
        'kernel_size': [3, 5],
        'stride': [1, 2],
        'padding': [0, 1],
        'dilation': [1, 2]
    }

    # train mode
    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x = torch.empty(0, in_cha, in_h, in_w)
        wrapper = Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x)

        # torch op with 3-dim input as shape reference
        x = torch.empty(3, in_cha, in_h, in_w)
        ref = nn.Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        ref_out = ref(x)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        x = torch.randn(3, in_cha, in_h, in_w)
        torch.manual_seed(0)
        wrapper = Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x)
        torch.manual_seed(0)
        ref = nn.Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
        ref_out = ref(x)
        assert torch.equal(wrapper_out, ref_out)

    # eval mode
    x = torch.empty(0, in_cha, in_h, in_w)
    wrapper = Conv2d(in_cha, out_cha, k, stride=s, padding=p, dilation=d)
    wrapper.eval()
    wrapper(x)


def test_conv_transposed_2d():
    test_cases = {
        'in_w': [10, 20],
        'in_h': [10, 20],
        'in_channel': [1, 3],
        'out_channel': [1, 3],
        'kernel_size': [3, 5],
        'stride': [1, 2],
        'padding': [0, 1],
        'dilation': [1, 2]
    }

    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x = torch.empty(0, in_cha, in_h, in_w)
        # out padding must be smaller than either stride or dilation
        op = min(s, d) - 1
        wrapper = ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        wrapper_out = wrapper(x)

        # torch op with 3-dim input as shape reference
        x = torch.empty(3, in_cha, in_h, in_w)
        ref = nn.ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        ref_out = ref(x)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        x = torch.randn(3, in_cha, in_h, in_w)
        torch.manual_seed(0)
        wrapper = ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        wrapper_out = wrapper(x)
        torch.manual_seed(0)
        ref = nn.ConvTranspose2d(
            in_cha,
            out_cha,
            k,
            stride=s,
            padding=p,
            dilation=d,
            output_padding=op)
        ref_out = ref(x)
        assert torch.equal(wrapper_out, ref_out)

    # eval mode
    x = torch.empty(0, in_cha, in_h, in_w)
    wrapper = ConvTranspose2d(
        in_cha, out_cha, k, stride=s, padding=p, dilation=d, output_padding=op)
    wrapper.eval()
    wrapper(x)


def test_max_pool_2d():
    test_cases = {
        'in_w': [10, 20],
        'in_h': [10, 20],
        'in_channel': [1, 3],
        'out_channel': [1, 3],
        'kernel_size': [3, 5],
        'stride': [1, 2],
        'padding': [0, 1],
        'dilation': [1, 2]
    }

    for in_h, in_w, in_cha, out_cha, k, s, p, d in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x = torch.empty(0, in_cha, in_h, in_w)
        wrapper = MaxPool2d(k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x)

        # torch op with 3-dim input as shape reference
        x = torch.empty(3, in_cha, in_h, in_w)
        ref = nn.MaxPool2d(k, stride=s, padding=p, dilation=d)
        ref_out = ref(x)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        x = torch.randn(3, in_cha, in_h, in_w)
        torch.manual_seed(0)
        wrapper = MaxPool2d(k, stride=s, padding=p, dilation=d)
        wrapper_out = wrapper(x)
        torch.manual_seed(0)
        ref = nn.MaxPool2d(k, stride=s, padding=p, dilation=d)
        ref_out = ref(x)
        assert torch.equal(wrapper_out, ref_out)


def test_linear():
    test_cases = {
        'in_w': [10, 20],
        'in_h': [10, 20],
        'in_feature': [1, 3],
        'out_feature': [1, 3]
    }

    for in_h, in_w, in_feature, out_feature in product(
            *list(test_cases.values())):
        # wrapper op with 0-dim input
        x = torch.empty(0, in_feature)
        wrapper = Linear(in_feature, out_feature)
        wrapper_out = wrapper(x)

        # torch op with 3-dim input as shape reference
        x = torch.empty(3, in_feature)
        ref = nn.Linear(in_feature, out_feature)
        ref_out = ref(x)

        assert wrapper_out.shape[0] == 0
        assert wrapper_out.shape[1:] == ref_out.shape[1:]

        wrapper_out.sum().backward()
        assert wrapper.weight.grad is not None
        assert wrapper.weight.grad.shape == wrapper.weight.shape

        x = torch.empty(3, in_feature)
        torch.manual_seed(0)
        wrapper = Linear(in_feature, out_feature)
        wrapper_out = wrapper(x)
        torch.manual_seed(0)
        ref = nn.Linear(in_feature, out_feature)
        ref_out = ref(x)
        assert torch.equal(wrapper_out, ref_out)

    # eval mode
    x = torch.empty(0, in_feature)
    wrapper = Linear(in_feature, out_feature)
    wrapper.eval()
    wrapper(x)


def test_nn_op_forward_called():
    torch.__version__ = '1.4.1'

    for m in ['Conv2d', 'ConvTranspose2d', 'MaxPool2d']:
        with patch('torch.nn.{}.forward'.format(m)) as nn_module_forward:
            # empty input
            x = torch.empty(0, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x)
            nn_module_forward.assert_called_with(x)

            # non-empty input
            x = torch.empty(1, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x)
            nn_module_forward.assert_called_with(x)

    with patch('torch.nn.Linear.forward') as nn_module_forward:
        # empty input
        x = torch.empty(0, 3)
        wrapper = Linear(3, 3)
        wrapper(x)
        nn_module_forward.assert_not_called()

        # non-empty input
        x = torch.empty(1, 3)
        wrapper = Linear(3, 3)
        wrapper(x)
        nn_module_forward.assert_called_with(x)
