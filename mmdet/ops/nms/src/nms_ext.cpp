// Modified from https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx, Soft-NMS is added
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

at::Tensor nms_cpu(const at::Tensor& dets, const float threshold);

at::Tensor soft_nms_cpu(const at::Tensor& dets, const float threshold,
                    const unsigned char method, const float sigma, const
                    float min_score);

std::vector<std::vector<int> > nms_match_cpu(const at::Tensor& dets, const float threshold);


#ifdef WITH_CUDA
at::Tensor nms_cuda(const at::Tensor& dets, const float threshold);
#endif

at::Tensor nms(const at::Tensor& dets, const float threshold){
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    return nms_cuda(dets, threshold);
#else
    AT_ERROR("nms is not compiled with GPU support");
#endif
  }
  return nms_cpu(dets, threshold);
}

at::Tensor soft_nms(const at::Tensor& dets, const float threshold,
                        const unsigned char method, const float sigma, const
                        float min_score) {
  if (dets.device().is_cuda()) {
    AT_ERROR("soft_nms is not implemented on GPU");
  }
  return soft_nms_cpu(dets, threshold, method, sigma, min_score);
}

std::vector<std::vector<int> > nms_match(const at::Tensor& dets, const float threshold) {
  if (dets.type().is_cuda()) {
    AT_ERROR("nms_match is not implemented on GPU");
  }
  return nms_match_cpu(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("soft_nms", &soft_nms, "soft non-maximum suppression");
  m.def("nms_match", &nms_match, "non-maximum suppression match");
}
