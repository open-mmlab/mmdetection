// Modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AffineGridGenerator.cpp
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/extension.h>

namespace mmdetection {

using namespace at;

Tensor affine_grid_generator_forward(const Tensor &theta, IntArrayRef size,
                                     bool align_corners);

Tensor affine_grid_generator_backward(const Tensor &grad, IntArrayRef size,
                                      bool align_corners);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("affine_grid_generator_forward", &affine_grid_generator_forward,
"affine_grid_generator_forward");
m.def("affine_grid_generator_backward", &affine_grid_generator_backward,
"affine_grid_generator_backward");
}

}  // namespace mmdetection
