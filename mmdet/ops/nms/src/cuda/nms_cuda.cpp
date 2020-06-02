// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor nms_cuda_forward(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor nms_cuda(const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda_forward(dets, threshold);
}
