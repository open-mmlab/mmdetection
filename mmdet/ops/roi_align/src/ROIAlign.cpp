// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Modified from https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlign

#pragma once
#include <torch/extension.h>
#include "ROIAlign.h"

namespace detectron2 {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlignV2 forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlignV2 backward");
}

} // namespace detectron2

