#include <torch/extension.h>

#include <cmath>
#include <vector>

int MaskedIm2colForwardLaucher(const at::Tensor im, const int height,
                               const int width, const int channels,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int mask_cnt,
                               at::Tensor col);

int MaskedCol2imForwardLaucher(const at::Tensor col, const int height,
                               const int width, const int channels,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int mask_cnt,
                               at::Tensor im);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int masked_im2col_forward_cuda(const at::Tensor im, const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int kernel_h,
                               const int kernel_w, const int pad_h,
                               const int pad_w, at::Tensor col) {
  CHECK_INPUT(im);
  CHECK_INPUT(mask_h_idx);
  CHECK_INPUT(mask_w_idx);
  CHECK_INPUT(col);
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)

  int channels = im.size(1);
  int height = im.size(2);
  int width = im.size(3);
  int mask_cnt = mask_h_idx.size(0);

  MaskedIm2colForwardLaucher(im, height, width, channels, kernel_h, kernel_w,
                             pad_h, pad_w, mask_h_idx, mask_w_idx, mask_cnt,
                             col);

  return 1;
}

int masked_col2im_forward_cuda(const at::Tensor col,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, int height,
                               int width, int channels, at::Tensor im) {
  CHECK_INPUT(col);
  CHECK_INPUT(mask_h_idx);
  CHECK_INPUT(mask_w_idx);
  CHECK_INPUT(im);
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)

  int mask_cnt = mask_h_idx.size(0);

  MaskedCol2imForwardLaucher(col, height, width, channels, mask_h_idx,
                             mask_w_idx, mask_cnt, im);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_im2col_forward", &masked_im2col_forward_cuda,
        "masked_im2col forward (CUDA)");
  m.def("masked_col2im_forward", &masked_col2im_forward_cuda,
        "masked_col2im forward (CUDA)");
}