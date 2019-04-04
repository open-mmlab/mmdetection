#include <torch/extension.h>

#include <cmath>
#include <vector>

int PSROIPoolForwardLauncher(const at::Tensor features, const at::Tensor rois,
                             const float spatial_scale, const int channels,
                             const int height, const int width,
                             const int num_rois, const int pooled_h,
                             const int pooled_w, const int group_size,
                             const int out_chn, at::Tensor top_data,
                             at::Tensor mapping_channel);

int PSROIPoolBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois,
                              const at::Tensor mapping_channel,
                              const float spatial_scale, const int channels,
                              const int height, const int width,
                              const int num_rois, const int pooled_h,
                              const int pooled_w, const int out_chn,
                              at::Tensor bottom_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int psroi_pooling_forward_cuda(at::Tensor features, at::Tensor rois,
                               int pooled_height, int pooled_width,
                               float spatial_scale, int group_size,
                               int out_channels, at::Tensor output,
                               at::Tensor mapping_channel) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);
  CHECK_INPUT(mapping_channel);

  // Get # of RoIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5) {
    return 0;
  }

  int in_channels = features.size(1);
  int in_height = features.size(2);
  int in_width = features.size(3);

  // call the gpu kernel for psroi_pooling
  PSROIPoolForwardLauncher(features, rois, spatial_scale, in_channels,
                           in_height, in_width, num_rois, pooled_height,
                           pooled_width, group_size, out_channels, output,
                           mapping_channel);
  return 1;
}

int psroi_pooling_backward_cuda(at::Tensor top_grad, at::Tensor rois,
                                at::Tensor mapping_channel, float spatial_scale,
                                at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rois);
  CHECK_INPUT(mapping_channel);
  CHECK_INPUT(bottom_grad);

  int out_channels = top_grad.size(1);
  int pooled_height = top_grad.size(2);
  int pooled_width = top_grad.size(3);

  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5) {
    return 0;
  }

  int in_channels = bottom_grad.size(1);
  int in_height = bottom_grad.size(2);
  int in_width = bottom_grad.size(3);

  PSROIPoolBackwardLauncher(top_grad, rois, mapping_channel, spatial_scale,
                            in_channels, in_height, in_width, num_rois,
                            pooled_width, pooled_height, out_channels,
                            bottom_grad);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &psroi_pooling_forward_cuda, "PSRoi_Pooling forward (CUDA)");
  m.def("backward", &psroi_pooling_backward_cuda,
        "PSRoi_Pooling backward (CUDA)");
}
