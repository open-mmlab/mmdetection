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

#ifdef WITH_CUDA
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
#endif

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_forward(const Tensor& input, const Tensor& grid,
                               int64_t interpolation_mode, int64_t padding_mode,
                               bool align_corners) {
    if (input.dim() == 4) {
        if (input.type().is_cuda()) {
#ifdef WITH_CUDA
            return grid_sampler_2d_forward_cuda(input, grid, interpolation_mode,
                padding_mode, align_corners);
#else
            AT_ERROR("grid_sampler is not compiled with GPU support");
#endif
        }
        return grid_sampler_2d_forward_cpu(input, grid, interpolation_mode,
                                           padding_mode, align_corners);
    } else {
        if (input.type().is_cuda()) {
#ifdef WITH_CUDA
            return grid_sampler_3d_forward_cuda(input, grid, interpolation_mode,
                padding_mode, align_corners);
#else
            AT_ERROR("grid_sampler is not compiled with GPU support");
#endif
        }
        return grid_sampler_3d_forward_cpu(input, grid, interpolation_mode,
                                           padding_mode, align_corners);
    }
}

std::tuple<Tensor, Tensor>
grid_sampler_backward(const Tensor& grad_output, const Tensor& input,
                         const Tensor& grid, int64_t interpolation_mode,
                         int64_t padding_mode, bool align_corners) {
    if (input.dim() == 4) {
        if (input.type().is_cuda()) {
#ifdef WITH_CUDA
            return grid_sampler_2d_backward_cuda(grad_output, input, grid,
                interpolation_mode,  padding_mode, align_corners);
#else
            AT_ERROR("grid_sampler is not compiled with GPU support");
#endif
        }
        return grid_sampler_2d_backward_cpu(grad_output, input, grid,
                                            interpolation_mode,  padding_mode, align_corners);
    } else {
        if (input.type().is_cuda()) {
#ifdef WITH_CUDA
            return grid_sampler_3d_backward_cuda(grad_output, input, grid,
                interpolation_mode,  padding_mode, align_corners);
#else
            AT_ERROR("grid_sampler is not compiled with GPU support");
#endif
        }
        return grid_sampler_3d_backward_cpu(grad_output, input, grid,
                                            interpolation_mode,  padding_mode, align_corners);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_sampler_forward_cuda", &grid_sampler_forward, "grid_sampler_forward");
  m.def("grid_sampler_backward_cuda", &grid_sampler_backward, "grid_sampler_backward");
}

}  // namespace mmdetection
