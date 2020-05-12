#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int masked_im2col_forward_cuda(const at::Tensor im, const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int kernel_h,
                               const int kernel_w, const int pad_h,
                               const int pad_w, at::Tensor col);

int masked_col2im_forward_cuda(const at::Tensor col,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, int height,
                               int width, int channels, at::Tensor im);
#endif

int masked_im2col_forward(const at::Tensor im, const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int kernel_h,
                               const int kernel_w, const int pad_h,
                               const int pad_w, at::Tensor col) {
  if (im.device().is_cuda()) {
#ifdef WITH_CUDA
    return masked_im2col_forward_cuda(im, mask_h_idx, mask_w_idx, kernel_h,
      kernel_w, pad_h, pad_w, col);
#else
    AT_ERROR("masked_im2col is not compiled with GPU support");
#endif
  }
  AT_ERROR("masked_im2col is not implemented on CPU");
}

int masked_col2im_forward(const at::Tensor col,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, int height,
                               int width, int channels, at::Tensor im) {
  if (col.device().is_cuda()) {
#ifdef WITH_CUDA
    return masked_col2im_forward_cuda(col, mask_h_idx, mask_w_idx, height,
      width, channels, im);
#else
    AT_ERROR("masked_col2im is not compiled with GPU support");
#endif
  }
  AT_ERROR("masked_col2im is not implemented on CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_im2col_forward", &masked_im2col_forward,
        "masked_im2col forward");
  m.def("masked_col2im_forward", &masked_col2im_forward,
        "masked_col2im forward");
}
