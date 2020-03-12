#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

namespace mmdetection {

using namespace at;

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_forward_cpu(const Tensor& input, const Tensor& grid,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_forward_cpu(const Tensor& input, const Tensor& grid,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode,
                              int64_t padding_mode, bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cpu(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_forward_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_forward_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode,
                              int64_t padding_mode, bool align_corners);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("grid_sampler_2d_forward_cpu", &grid_sampler_2d_forward_cpu, "grid_sampler_2d_forward (CPU)");
  m.def("grid_sampler_2d_backward_cpu", &grid_sampler_2d_backward_cpu, "grid_sampler_2d_backward (CPU)");
  m.def("grid_sampler_3d_forward_cpu", &grid_sampler_3d_forward_cpu, "grid_sampler_3d_forward (CPU)");
  m.def("grid_sampler_3d_backward_cpu", &grid_sampler_3d_backward_cpu, "grid_sampler_3d_backward (CPU)");

  m.def("grid_sampler_2d_forward_cuda", &grid_sampler_2d_forward_cuda, "grid_sampler_2d_forward (CUDA)");
  m.def("grid_sampler_2d_backward_cuda", &grid_sampler_2d_backward_cuda, "grid_sampler_2d_backward (CUDA)");
  m.def("grid_sampler_3d_forward_cuda", &grid_sampler_3d_forward_cuda, "grid_sampler_3d_forward (CUDA)");
  m.def("grid_sampler_3d_backward_cuda", &grid_sampler_3d_backward_cuda, "grid_sampler_3d_backward (CUDA)");
}

}  // namespace mmdetection
