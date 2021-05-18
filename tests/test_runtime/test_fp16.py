import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.fp16_utils import cast_tensor_type


def test_cast_tensor_type():
    inputs = torch.FloatTensor([5.])
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dtype == dst_type

    inputs = 'tensor'
    src_type = str
    dst_type = str
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, str)

    inputs = np.array([5.])
    src_type = np.ndarray
    dst_type = np.ndarray
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, np.ndarray)

    inputs = dict(
        tensor_a=torch.FloatTensor([1.]), tensor_b=torch.FloatTensor([2.]))
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, dict)
    assert outputs['tensor_a'].dtype == dst_type
    assert outputs['tensor_b'].dtype == dst_type

    inputs = [torch.FloatTensor([1.]), torch.FloatTensor([2.])]
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, list)
    assert outputs[0].dtype == dst_type
    assert outputs[1].dtype == dst_type

    inputs = 5
    outputs = cast_tensor_type(inputs, None, None)
    assert isinstance(outputs, int)


def test_auto_fp16():

    with pytest.raises(TypeError):
        # ExampleObject is not a subclass of nn.Module

        class ExampleObject:

            @auto_fp16()
            def __call__(self, x):
                return x

        model = ExampleObject()
        input_x = torch.ones(1, dtype=torch.float32)
        model(input_x)

    # apply to all input args
    class ExampleModule(nn.Module):

        @auto_fp16()
        def forward(self, x, y):
            return x, y

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.float32)
    input_y = torch.ones(1, dtype=torch.float32)
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32

    model.fp16_enabled = True
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y = model(input_x.cuda(), input_y.cuda())
        assert output_x.dtype == torch.half
        assert output_y.dtype == torch.half

    # apply to specified input args
    class ExampleModule(nn.Module):

        @auto_fp16(apply_to=('x', ))
        def forward(self, x, y):
            return x, y

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.float32)
    input_y = torch.ones(1, dtype=torch.float32)
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32

    model.fp16_enabled = True
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.float32

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y = model(input_x.cuda(), input_y.cuda())
        assert output_x.dtype == torch.half
        assert output_y.dtype == torch.float32

    # apply to optional input args
    class ExampleModule(nn.Module):

        @auto_fp16(apply_to=('x', 'y'))
        def forward(self, x, y=None, z=None):
            return x, y, z

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.float32)
    input_y = torch.ones(1, dtype=torch.float32)
    input_z = torch.ones(1, dtype=torch.float32)
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32
    assert output_z.dtype == torch.float32

    model.fp16_enabled = True
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half
    assert output_z.dtype == torch.float32

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y, output_z = model(
            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
        assert output_x.dtype == torch.half
        assert output_y.dtype == torch.half
        assert output_z.dtype == torch.float32

    # out_fp32=True
    class ExampleModule(nn.Module):

        @auto_fp16(apply_to=('x', 'y'), out_fp32=True)
        def forward(self, x, y=None, z=None):
            return x, y, z

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.half)
    input_y = torch.ones(1, dtype=torch.float32)
    input_z = torch.ones(1, dtype=torch.float32)
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.float32
    assert output_z.dtype == torch.float32

    model.fp16_enabled = True
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32
    assert output_z.dtype == torch.float32

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y, output_z = model(
            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
        assert output_x.dtype == torch.float32
        assert output_y.dtype == torch.float32
        assert output_z.dtype == torch.float32


def test_force_fp32():

    with pytest.raises(TypeError):
        # ExampleObject is not a subclass of nn.Module

        class ExampleObject:

            @force_fp32()
            def __call__(self, x):
                return x

        model = ExampleObject()
        input_x = torch.ones(1, dtype=torch.float32)
        model(input_x)

    # apply to all input args
    class ExampleModule(nn.Module):

        @force_fp32()
        def forward(self, x, y):
            return x, y

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.half)
    input_y = torch.ones(1, dtype=torch.half)
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half

    model.fp16_enabled = True
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y = model(input_x.cuda(), input_y.cuda())
        assert output_x.dtype == torch.float32
        assert output_y.dtype == torch.float32

    # apply to specified input args
    class ExampleModule(nn.Module):

        @force_fp32(apply_to=('x', ))
        def forward(self, x, y):
            return x, y

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.half)
    input_y = torch.ones(1, dtype=torch.half)
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half

    model.fp16_enabled = True
    output_x, output_y = model(input_x, input_y)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.half

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y = model(input_x.cuda(), input_y.cuda())
        assert output_x.dtype == torch.float32
        assert output_y.dtype == torch.half

    # apply to optional input args
    class ExampleModule(nn.Module):

        @force_fp32(apply_to=('x', 'y'))
        def forward(self, x, y=None, z=None):
            return x, y, z

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.half)
    input_y = torch.ones(1, dtype=torch.half)
    input_z = torch.ones(1, dtype=torch.half)
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half
    assert output_z.dtype == torch.half

    model.fp16_enabled = True
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.float32
    assert output_z.dtype == torch.half

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y, output_z = model(
            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
        assert output_x.dtype == torch.float32
        assert output_y.dtype == torch.float32
        assert output_z.dtype == torch.half

    # out_fp16=True
    class ExampleModule(nn.Module):

        @force_fp32(apply_to=('x', 'y'), out_fp16=True)
        def forward(self, x, y=None, z=None):
            return x, y, z

    model = ExampleModule()
    input_x = torch.ones(1, dtype=torch.float32)
    input_y = torch.ones(1, dtype=torch.half)
    input_z = torch.ones(1, dtype=torch.half)
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.float32
    assert output_y.dtype == torch.half
    assert output_z.dtype == torch.half

    model.fp16_enabled = True
    output_x, output_y, output_z = model(input_x, y=input_y, z=input_z)
    assert output_x.dtype == torch.half
    assert output_y.dtype == torch.half
    assert output_z.dtype == torch.half

    if torch.cuda.is_available():
        model.cuda()
        output_x, output_y, output_z = model(
            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
        assert output_x.dtype == torch.half
        assert output_y.dtype == torch.half
        assert output_z.dtype == torch.half
