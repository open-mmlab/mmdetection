// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/modulated_dcn_cuda.c

// based on
// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
void deform_psroi_pooling_cuda_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, at::Tensor out,
    at::Tensor top_count, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void deform_psroi_pooling_cuda_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, at::Tensor trans,
    at::Tensor top_count, at::Tensor input_grad, at::Tensor trans_grad,
    const int no_trans, const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std);
#endif

void deform_psroi_pooling_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, at::Tensor out,
    at::Tensor top_count, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_forward(input, bbox, trans, out, top_count,
        no_trans, spatial_scale, output_dim, group_size, pooled_size,
        part_size, sample_per_part, trans_std);
#else
    AT_ERROR("deform psroi pooling is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform psroi pooling is not implemented on CPU");
}

void deform_psroi_pooling_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, at::Tensor trans,
    at::Tensor top_count, at::Tensor input_grad, at::Tensor trans_grad,
    const int no_trans, const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_backward(out_grad, input, bbox, trans,
        top_count, input_grad, trans_grad, no_trans, spatial_scale,
        output_dim, group_size, pooled_size, part_size, sample_per_part,
        trans_std);
#else
    AT_ERROR("deform psroi pooling is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform psroi pooling is not implemented on CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward,
        "deform psroi pooling forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward,
        "deform psroi pooling backward");
}
