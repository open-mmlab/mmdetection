#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65536;
  return min(optimal_block_num, max_block_num);
}

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}
template <typename scalar_t>
__global__ void CARAFENAIVEForward(const int nthreads,
                                   const scalar_t *bottom_data,
                                   const scalar_t *bottom_masks,
                                   const int kernel_size, const int group_size,
                                   const int scale_factor, const int channels,
                                   const int height, const int width,
                                   scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size * group_size;
    int mask_group = c / (channels / group_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
    int start_w = down_pw - (kernel_size - 1) / 2;
    int end_w = down_pw + (kernel_size - 1) / 2 + 1;
    int start_h = down_ph - (kernel_size - 1) / 2;
    int end_h = down_ph + (kernel_size - 1) / 2 + 1;

    scalar_t output_val = 0;
    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, c, iy, ix, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        output_val += bottom_data[feat_index] * bottom_masks[mask_index];
      }
    }
    top_data[index] = output_val;
  }
}

int CARAFENAIVEForwardLaucher(const at::Tensor features, const at::Tensor masks,
                              const int kernel_size, const int group_size,
                              const int scale_factor, const int batch_size,
                              const int channels, const int height,
                              const int width, at::Tensor output) {
  const int output_size = batch_size * channels * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "CARAFENAIVELaucherForward", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        CARAFENAIVEForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, bottom_masks, kernel_size, group_size,
                scale_factor, channels, height, width, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void CARAFENAIVEBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_data,
    const scalar_t *bottom_masks, const int kernel_size, const int group_size,
    const int scale_factor, const int channels, const int height,
    const int width, scalar_t *bottom_diff, scalar_t *mask_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size * group_size;
    int mask_group = c / (channels / group_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
    int start_w = down_pw - (kernel_size - 1) / 2;
    int end_w = down_pw + (kernel_size - 1) / 2 + 1;
    int start_h = down_ph - (kernel_size - 1) / 2;
    int end_h = down_ph + (kernel_size - 1) / 2 + 1;

    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, c, iy, ix, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        atomicAdd(bottom_diff + feat_index,
                  bottom_masks[mask_index] * top_diff[index]);
        atomicAdd(mask_diff + mask_index,
                  bottom_data[feat_index] * top_diff[index]);
      }
    }
  }
}

int CARAFENAIVEBackwardLaucher(const at::Tensor top_grad,
                               const at::Tensor features,
                               const at::Tensor masks, const int kernel_size,
                               const int group_size, const int scale_factor,
                               const int batch_size, const int channels,
                               const int height, const int width,
                               at::Tensor bottom_grad, at::Tensor mask_grad) {
  const int output_size = batch_size * channels * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "CARAFENAIVELaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        scalar_t *mask_diff = mask_grad.data_ptr<scalar_t>();

        CARAFENAIVEBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, bottom_data, bottom_masks, kernel_size,
                group_size, scale_factor, channels, height, width, bottom_diff,
                mask_diff);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
