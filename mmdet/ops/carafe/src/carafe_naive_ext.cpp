#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int carafe_naive_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int group_size, int scale_factor,
                              at::Tensor output);

int carafe_naive_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int group_size, int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad);
#endif

int carafe_naive_forward(at::Tensor features, at::Tensor masks,
                         int kernel_size, int group_size, int scale_factor,
                         at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return carafe_naive_forward_cuda(features, masks, kernel_size,
        group_size, scale_factor, output);
#else
    AT_ERROR("carafe naive is not compiled with GPU support");
#endif
  }
  AT_ERROR("carafe naive is not implemented on CPU");
}

int carafe_naive_backward(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int group_size, int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return carafe_naive_backward_cuda(top_grad, features, masks, kernel_size,
        group_size, scale_factor, bottom_grad, mask_grad);
#else
    AT_ERROR("carafe naive is not compiled with GPU support");
#endif
  }
  AT_ERROR("carafe naive is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &carafe_naive_forward, "carafe_naive forward");
  m.def("backward", &carafe_naive_backward, "carafe_naive backward");
}
