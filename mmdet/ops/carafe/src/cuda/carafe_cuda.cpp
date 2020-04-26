#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

int CARAFEForwardLaucher(const at::Tensor features, const at::Tensor masks,
                         const int kernel_size, const int group_size,
                         const int scale_factor, const int batch_size,
                         const int channels, const int input_height,
                         const int input_width, const int output_height,
                         const int output_width, const int mask_channels,
                         at::Tensor rfeatures, at::Tensor routput,
                         at::Tensor rmasks, at::Tensor output);

int CARAFEBackwardLaucher(const at::Tensor top_grad, const at::Tensor rfeatures,
                          const at::Tensor masks, const int kernel_size,
                          const int group_size, const int scale_factor,
                          const int batch_size, const int channels,
                          const int input_height, const int input_width,
                          const int output_height, const int output_width,
                          const int mask_channels, at::Tensor rtop_grad,
                          at::Tensor rbottom_grad_hs, at::Tensor rbottom_grad,
                          at::Tensor rmask_grad, at::Tensor bottom_grad,
                          at::Tensor mask_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int carafe_forward_cuda(at::Tensor features, at::Tensor rfeatures,
                        at::Tensor masks, at::Tensor rmasks, int kernel_size,
                        int group_size, int scale_factor, at::Tensor routput,
                        at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rfeatures);
  CHECK_INPUT(masks);
  CHECK_INPUT(rmasks);
  CHECK_INPUT(output);
  CHECK_INPUT(routput);
  at::DeviceGuard guard(features.device());

  const int batch_size = output.size(0);
  const int num_channels = output.size(1);
  const int output_height = output.size(2);
  const int output_width = output.size(3);

  const int input_height = features.size(2);
  const int input_width = features.size(3);

  const int mask_channels = masks.size(1);

  rfeatures.resize_({batch_size, input_height, input_width, num_channels});
  routput.resize_({batch_size, output_height, output_width, num_channels});
  rmasks.resize_({batch_size, output_height, output_width, mask_channels});

  CARAFEForwardLaucher(features, masks, kernel_size, group_size, scale_factor,
                       batch_size, num_channels, input_height, input_width,
                       output_height, output_width, mask_channels, rfeatures,
                       routput, rmasks, output);

  return 1;
}

int carafe_backward_cuda(at::Tensor top_grad, at::Tensor rfeatures,
                         at::Tensor masks, int kernel_size, int group_size,
                         int scale_factor, at::Tensor rtop_grad,
                         at::Tensor rbottom_grad_hs, at::Tensor rbottom_grad,
                         at::Tensor rmask_grad, at::Tensor bottom_grad,
                         at::Tensor mask_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rfeatures);
  CHECK_INPUT(masks);
  CHECK_INPUT(rtop_grad);
  CHECK_INPUT(rbottom_grad_hs);
  CHECK_INPUT(rbottom_grad);
  CHECK_INPUT(rmask_grad);
  CHECK_INPUT(bottom_grad);
  CHECK_INPUT(mask_grad);
  at::DeviceGuard guard(top_grad.device());

  const int batch_size = top_grad.size(0);
  const int num_channels = top_grad.size(1);
  const int output_height = top_grad.size(2);
  const int output_width = top_grad.size(3);

  const int input_height = bottom_grad.size(2);
  const int input_width = bottom_grad.size(3);

  const int mask_channels = masks.size(1);

  rtop_grad.resize_({batch_size, output_height, output_width, num_channels});
  rbottom_grad.resize_({batch_size, input_height, input_width, num_channels});
  rbottom_grad_hs.resize_(
      {batch_size, output_height, output_width, num_channels});
  rmask_grad.resize_({batch_size, output_height, output_width, mask_channels});

  CARAFEBackwardLaucher(top_grad, rfeatures, masks, kernel_size, group_size,
                        scale_factor, batch_size, num_channels, input_height,
                        input_width, output_height, output_width, mask_channels,
                        rtop_grad, rbottom_grad_hs, rbottom_grad, rmask_grad,
                        bottom_grad, mask_grad);

  return 1;
}
