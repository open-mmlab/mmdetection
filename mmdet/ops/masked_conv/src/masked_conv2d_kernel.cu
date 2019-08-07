#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void MaskedIm2colForward(const int n, const scalar_t *data_im,
                                    const int height, const int width,
                                    const int kernel_h, const int kernel_w,
                                    const int pad_h, const int pad_w,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_col) {
  // mask_cnt * channels
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_col = mask_h_idx[m_index];
    const int w_col = mask_w_idx[m_index];
    const int c_im = index / mask_cnt;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col - pad_h;
    const int w_offset = w_col - pad_w;
    scalar_t *data_col_ptr = data_col + c_col * mask_cnt + m_index;
    for (int i = 0; i < kernel_h; ++i) {
      int h_im = h_offset + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_im = w_offset + j;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          *data_col_ptr =
              (scalar_t)data_im[(c_im * height + h_im) * width + w_im];
        } else {
          *data_col_ptr = 0.0;
        }
        data_col_ptr += mask_cnt;
      }
    }
  }
}

int MaskedIm2colForwardLaucher(const at::Tensor bottom_data, const int height,
                               const int width, const int channels,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int mask_cnt,
                               at::Tensor top_data) {
  const int output_size = mask_cnt * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedIm2colLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data<int64_t>();
        scalar_t *top_data_ = top_data.data<scalar_t>();
        MaskedIm2colForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data_, height, width, kernel_h, kernel_w,
                pad_h, pad_w, mask_h_idx_, mask_w_idx_, mask_cnt, top_data_);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__global__ void MaskedCol2imForward(const int n, const scalar_t *data_col,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_im = mask_h_idx[m_index];
    const int w_im = mask_w_idx[m_index];
    const int c_im = index / mask_cnt;
    // compute the start and end of the output
    data_im[(c_im * height + h_im) * width + w_im] = data_col[index];
  }
}

int MaskedCol2imForwardLaucher(const at::Tensor bottom_data, const int height,
                               const int width, const int channels,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int mask_cnt,
                               at::Tensor top_data) {
  const int output_size = mask_cnt * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedCol2imLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data<int64_t>();
        scalar_t *top_data_ = top_data.data<scalar_t>();

        MaskedCol2imForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data_, height, width, channels, mask_h_idx_,
                mask_w_idx_, mask_cnt, top_data_);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
