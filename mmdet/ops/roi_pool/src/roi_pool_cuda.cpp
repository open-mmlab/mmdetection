#include <torch/extension.h>

#include <cmath>
#include <vector>

int ROIPoolForwardLaucher(const at::Tensor features, const at::Tensor rois,
                          const float spatial_scale, const int channels,
                          const int height, const int width, const int num_rois,
                          const int pooled_h, const int pooled_w,
                          at::Tensor output, at::Tensor argmax);

int ROIPoolBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                           const at::Tensor argmax, const float spatial_scale,
                           const int batch_size, const int channels,
                           const int height, const int width,
                           const int num_rois, const int pooled_h,
                           const int pooled_w, at::Tensor bottom_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int roi_pooling_forward_cuda(at::Tensor features, at::Tensor rois,
                             int pooled_height, int pooled_width,
                             float spatial_scale, at::Tensor output,
                             at::Tensor argmax) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);
  CHECK_INPUT(argmax);
  at::DeviceGuard guard(features.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int channels = features.size(1);
  int height = features.size(2);
  int width = features.size(3);

  ROIPoolForwardLaucher(features, rois, spatial_scale, channels, height, width,
                        num_rois, pooled_height, pooled_width, output, argmax);

  return 1;
}

int roi_pooling_backward_cuda(at::Tensor top_grad, at::Tensor rois,
                              at::Tensor argmax, float spatial_scale,
                              at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rois);
  CHECK_INPUT(argmax);
  CHECK_INPUT(bottom_grad);
  at::DeviceGuard guard(top_grad.device());

  int pooled_height = top_grad.size(2);
  int pooled_width = top_grad.size(3);
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }
  int batch_size = bottom_grad.size(0);
  int channels = bottom_grad.size(1);
  int height = bottom_grad.size(2);
  int width = bottom_grad.size(3);

  ROIPoolBackwardLaucher(top_grad, rois, argmax, spatial_scale, batch_size,
                         channels, height, width, num_rois, pooled_height,
                         pooled_width, bottom_grad);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_pooling_forward_cuda, "Roi_Pooling forward (CUDA)");
  m.def("backward", &roi_pooling_backward_cuda, "Roi_Pooling backward (CUDA)");
}
