// modify from
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/SigmoidFocalLoss.h
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                         const at::Tensor &targets,
                                         const int num_classes,
                                         const float gamma, const float alpha);

at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                          const at::Tensor &targets,
                                          const at::Tensor &d_losses,
                                          const int num_classes,
                                          const float gamma, const float alpha);
#endif

// Interface for Python
at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
                                    const at::Tensor &targets,
                                    const int num_classes, const float gamma,
                                    const float alpha) {
  if (logits.device().is_cuda()) {
#ifdef WITH_CUDA
    at::DeviceGuard guard(logits.device());
    return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma,
                                         alpha);
#else
      AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
  }
  AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
                                     const at::Tensor &targets,
                                     const at::Tensor &d_losses,
                                     const int num_classes, const float gamma,
                                     const float alpha) {
  if (logits.device().is_cuda()) {
#ifdef WITH_CUDA
    at::DeviceGuard guard(logits.device());
    return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses,
                                          num_classes, gamma, alpha);
#else
      AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
  }
  AT_ERROR("SigmoidFocalLoss is not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SigmoidFocalLoss_forward,
        "SigmoidFocalLoss forward");
  m.def("backward", &SigmoidFocalLoss_backward,
        "SigmoidFocalLoss backward");
}
