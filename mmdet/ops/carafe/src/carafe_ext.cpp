#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int carafe_forward_cuda(at::Tensor features, at::Tensor rfeatures,
                        at::Tensor masks, at::Tensor rmasks, int kernel_size,
                        int group_size, int scale_factor, at::Tensor routput,
                        at::Tensor output);

int carafe_backward_cuda(at::Tensor top_grad, at::Tensor rfeatures,
                         at::Tensor masks, int kernel_size, int group_size,
                         int scale_factor, at::Tensor rtop_grad,
                         at::Tensor rbottom_grad_hs, at::Tensor rbottom_grad,
                         at::Tensor rmask_grad, at::Tensor bottom_grad,
                         at::Tensor mask_grad);
#endif

int carafe_forward(at::Tensor features, at::Tensor rfeatures,
                   at::Tensor masks, at::Tensor rmasks, int kernel_size,
                   int group_size, int scale_factor, at::Tensor routput,
                   at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return carafe_forward_cuda(features, rfeatures, masks, rmasks, kernel_size,
                               group_size, scale_factor, routput, output);
#else
    AT_ERROR("carafe is not compiled with GPU support");
#endif
  }
  AT_ERROR("carafe is not implemented on CPU");
}

int carafe_backward(at::Tensor top_grad, at::Tensor rfeatures,
                    at::Tensor masks, int kernel_size, int group_size,
                    int scale_factor, at::Tensor rtop_grad,
                    at::Tensor rbottom_grad_hs, at::Tensor rbottom_grad,
                    at::Tensor rmask_grad, at::Tensor bottom_grad,
                    at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return carafe_backward_cuda(top_grad, rfeatures, masks, kernel_size,
        group_size, scale_factor, rtop_grad, rbottom_grad_hs, rbottom_grad,
        rmask_grad, bottom_grad, mask_grad);
#else
    AT_ERROR("carafe is not compiled with GPU support");
#endif
  }
  AT_ERROR("carafe is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &carafe_forward, "carafe forward");
  m.def("backward", &carafe_backward, "carafe backward");
}
