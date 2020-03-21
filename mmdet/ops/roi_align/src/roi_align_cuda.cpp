#include <torch/extension.h>

#include <ATen/ATen.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int ROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output);

int ROIAlignBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor bottom_grad);

at::Tensor ROIAlignForwardV2Laucher(const at::Tensor& input,
                                    const at::Tensor& rois,
                                    const float spatial_scale,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int sampling_ratio, bool aligned);

at::Tensor ROIAlignBackwardV2Laucher(
    const at::Tensor& grad, const at::Tensor& rois, const float spatial_scale,
    const int pooled_height, const int pooled_width, const int batch_size,
    const int channels, const int height, const int width,
    const int sampling_ratio, bool aligned);
#endif

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int ROIAlign_forwardV1(at::Tensor features, at::Tensor rois, int pooled_height,
                       int pooled_width, float spatial_scale, int sample_num,
                       at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignForwardLaucher(features, rois, spatial_scale, sample_num,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height, pooled_width, output);

  return 1;
}

int ROIAlign_backwardV1(at::Tensor top_grad, at::Tensor rois, int pooled_height,
                        int pooled_width, float spatial_scale, int sample_num,
                        at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rois);
  CHECK_INPUT(bottom_grad);
  at::DeviceGuard guard(top_grad.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);

  ROIAlignBackwardLaucher(top_grad, rois, spatial_scale, sample_num,
                          num_channels, data_height, data_width, num_rois,
                          pooled_height, pooled_width, bottom_grad);

  return 1;
}

// Interface for Python
inline at::Tensor ROIAlign_forwardV2(const at::Tensor& input,
                                     const at::Tensor& rois,
                                     const float spatial_scale,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int sampling_ratio, bool aligned) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignForwardV2Laucher(input, rois, spatial_scale, pooled_height,
                                    pooled_width, sampling_ratio, aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
}

inline at::Tensor ROIAlign_backwardV2(
    const at::Tensor& grad, const at::Tensor& rois, const float spatial_scale,
    const int pooled_height, const int pooled_width, const int batch_size,
    const int channels, const int height, const int width,
    const int sampling_ratio, bool aligned) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignBackwardV2Laucher(grad, rois, spatial_scale, pooled_height,
                                     pooled_width, batch_size, channels, height,
                                     width, sampling_ratio, aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_v1", &ROIAlign_forwardV1, "Roi_Align V1 forward (CUDA)");
  m.def("backward_v1", &ROIAlign_backwardV1, "Roi_Align V1 backward (CUDA)");
  m.def("forward_v2", &ROIAlign_forwardV2, "Roi_Align V2 forward (CUDA)");
  m.def("backward_v2", &ROIAlign_backwardV2, "Roi_Align V2 backward (CUDA)");
}
