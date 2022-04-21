import numpy as np
import torch

from mmdet.utils import AvoidOOM


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
