import numpy as np
import pytest
import torch

from mmdet.utils import AvoidOOM
from mmdet.utils.memory import convert_tensor_type


def test_avoidoom():
    tensor = torch.from_numpy(np.random.random((20, 20)))
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        # get default result
        default_result = torch.mm(tensor, tensor.transpose(1, 0))

        # when not occurred OOM error
        AvoidCudaOOM = AvoidOOM()
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert default_result.device == result.device and \
               default_result.dtype == result.dtype and \
               torch.equal(default_result, result)

        # calculate with fp16 and convert back to source type
        AvoidCudaOOM = AvoidOOM(test=True)
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert default_result.device == result.device and \
               default_result.dtype == result.dtype and \
               torch.allclose(default_result, result, 1e-3)

        # calculate with fp16 and do not convert back to source type
        AvoidCudaOOM = AvoidOOM(keep_type=False, test=True)
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert result.dtype == torch.half and \
               result.device == default_result.device

        # calculate on cpu and convert back to source device
        AvoidCudaOOM = AvoidOOM(keep_type=False, test=True, return_fp16=False)
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert result.dtype == default_result.dtype and \
               result.device == default_result.device and \
               torch.allclose(default_result, result)

        # calculate on cpu and do not convert back to source device
        AvoidCudaOOM = AvoidOOM(
            keep_type=False, test=True, return_fp16=False, return_gpu=False)
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        cpu_device = torch.empty(0).device
        assert result.dtype == default_result.dtype and \
               result.device == cpu_device and \
               torch.allclose(default_result.cpu(), result)

        # do not calculate on cpu and the outputs will be fp16
        AvoidCudaOOM = AvoidOOM(
            keep_type=False,
            test=True,
            return_fp16=False,
            return_gpu=False,
            convert_cpu=False)
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert result.dtype == torch.half and \
               result.device == default_result.device

    else:
        default_result = torch.mm(tensor, tensor.transpose(1, 0))
        AvoidCudaOOM = AvoidOOM()
        result = AvoidCudaOOM.retry_if_cuda_oom(torch.mm)(tensor,
                                                          tensor.transpose(
                                                              1, 0))
        assert default_result.device == result.device and \
               default_result.dtype == result.dtype and \
               torch.equal(default_result, result)


def test_conver_tensor_type():
    inputs = torch.rand(10)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    with pytest.raises(AssertionError):
        convert_tensor_type(inputs, src_type=None, dst_type=None)
    # input is a float
    out = convert_tensor_type(10., dst_type=torch.half)
    assert out == 10. and isinstance(out, float)
    # convert Tensor to fp16 and re-convert to fp32
    fp16_out = convert_tensor_type(inputs, dst_type=torch.half)
    assert fp16_out.dtype == torch.half
    fp32_out = convert_tensor_type(fp16_out, dst_type=torch.float32)
    assert fp32_out.dtype == torch.float32

    # input is a list
    list_input = [inputs, inputs]
    list_outs = convert_tensor_type(list_input, dst_type=torch.half)
    assert len(list_outs) == len(list_input) and \
           isinstance(list_outs, list)
    for out in list_outs:
        assert out.dtype == torch.half
    # input is a dict
    dict_input = {'test1': inputs, 'test2': inputs}
    dict_outs = convert_tensor_type(dict_input, dst_type=torch.half)
    assert len(dict_outs) == len(dict_input) and \
           isinstance(dict_outs, dict)

    # convert the input tensor to CPU and re-convert to GPU
    if torch.cuda.is_available():
        cpu_device = torch.empty(0).device
        gpu_device = inputs.device
        cpu_out = convert_tensor_type(inputs, dst_type=cpu_device)
        assert cpu_out.device == cpu_device

        gpu_out = convert_tensor_type(inputs, dst_type=gpu_device)
        assert gpu_out.device == gpu_device
