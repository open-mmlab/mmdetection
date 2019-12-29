#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace at;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024  // 32 * 32
#define WARP_SIZE 32
#define THREADS_PER_PIXEL 32
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144
#define MAXIMIZE_KERNEL_SIZE true
#define kTileDim 32
#define kBlockRows 8
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}
/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

// Splits the original matrix into submatrices with size 32 * 32.
// Each block transposes one submatrix by loading it into shared memory.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename scalar_t>
__global__ void BatchTranspose2DCUDAKernel(const int N, const int H,
                                           const int W, const int dh,
                                           const int dw,
                                           const scalar_t *__restrict__ X,
                                           scalar_t *__restrict__ Y) {
  __shared__ scalar_t tile[kTileDim][kTileDim + 1];
  const int n = blockIdx.x / (dh * dw);
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = n * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
      tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}
template <typename scalar_t>
__global__ void CARAFEForward(
    const int num_kernels, const scalar_t *__restrict__ bottom_data,
    const scalar_t *__restrict__ bottom_masks, const int kernel_size,
    const int group_size, const int scale_factor, const int channels,
    const int down_height, const int down_width, const int height,
    const int width, const int mask_channels, scalar_t *__restrict__ top_data) {
#if MAXIMIZE_KERNEL_SIZE
  __shared__ float shared_mask[MAX_SHARED_SCALAR_T * 2];
#else
  __shared__ scalar_t shared_mask[MAX_SHARED_SCALAR_T];
#endif

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }
  const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int down_pw = pw / scale_factor;
  const int down_ph = ph / scale_factor;

  const int start_w = down_pw - (kernel_size - 1) / 2;
  const int end_w = down_pw + (kernel_size - 1) / 2 + 1;
  const int start_h = down_ph - (kernel_size - 1) / 2;
  const int end_h = down_ph + (kernel_size - 1) / 2 + 1;
  for (int c = split_id; c < mask_channels; c += THREADS_PER_PIXEL) {
    int mask_index = Loc2Index(n, ph, pw, c, height, width, mask_channels);
    shared_mask[c * WARP_SIZE + pixel_id] = bottom_masks[mask_index];
  }
  __syncthreads();

  const int channels_per_group = ceilf(channels / (float)group_size);
#pragma unroll
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    int mask_group = c / channels_per_group;
    scalar_t output_val = 0;
#pragma unroll
    for (int iy = start_h; iy < end_h; iy++) {
#pragma unroll
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, iy, ix, c, down_height, down_width, channels);

        output_val += bottom_data[feat_index] *
                      shared_mask[mask_c * WARP_SIZE + pixel_id];
      }
    }

    int top_index = Loc2Index(n, ph, pw, c, height, width, channels);
    top_data[top_index] = output_val;
  }
}

int CARAFEForwardLaucher(const at::Tensor features, const at::Tensor masks,
                         const int kernel_size, const int group_size,
                         const int scale_factor, const int batch_size,
                         const int channels, const int input_height,
                         const int input_width, const int output_height,
                         const int output_width, const int mask_channels,
                         at::Tensor rfeatures, at::Tensor routput,
                         at::Tensor rmasks, at::Tensor output) {
  // one warp per pixel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "NCHW2NHWC_Feature", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        scalar_t *top_data = rfeatures.data<scalar_t>();
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(input_height * input_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, input_height * input_width, dh, dw,
                bottom_data, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "NCHW2NHWC_Masks", ([&] {
        const scalar_t *bottom_data = masks.data<scalar_t>();
        scalar_t *top_data = rmasks.data<scalar_t>();
        const int dh = divideUP(mask_channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, mask_channels, output_height * output_width, dh, dw,
                bottom_data, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "CARAFELaucherForward", ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;
        const scalar_t *bottom_data = rfeatures.data<scalar_t>();
        const scalar_t *bottom_masks = rmasks.data<scalar_t>();
        scalar_t *top_data = routput.data<scalar_t>();

        CARAFEForward<scalar_t>
            <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
               THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, bottom_data, bottom_masks, kernel_size, group_size,
                scale_factor, channels, input_height, input_width,
                output_height, output_width, mask_channels, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "NHWC2NCHW", ([&] {
        const scalar_t *bottom_data = routput.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, channels, dh, dw,
                bottom_data, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void CARAFEBackward_Feature(
    const int num_kernels, const scalar_t *__restrict__ top_diff,
    const scalar_t *__restrict__ bottom_masks, const int kernel_size,
    const int group_size, const int scale_factor, const int channels,
    const int down_height, const int down_width, const int height,
    const int width, const int mask_channels,
    scalar_t *__restrict__ bottom_diff) {
#if MAXIMIZE_KERNEL_SIZE
  __shared__ float shared_mask[MAX_SHARED_SCALAR_T * 2];
#else
  __shared__ scalar_t shared_mask[MAX_SHARED_SCALAR_T];
#endif

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }

  const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  // (n, c, ph, pw) is an element in the bottom_data
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int start_w = pw - (kernel_size - 1) * scale_factor / 2;
  const int end_w = pw + (kernel_size - 1) * scale_factor / 2 + 1;
  const int start_h = ph - (kernel_size - 1) * scale_factor / 2;
  const int end_h = ph + (kernel_size - 1) * scale_factor / 2 + 1;
  for (int c = split_id; c < mask_channels; c += THREADS_PER_PIXEL) {
    const int mask_w = (c % kernel_size) * scale_factor;
    const int mask_h = (c / kernel_size % kernel_size) * scale_factor;
    const int mask_x = start_w + mask_w;
    const int mask_y = start_h + mask_h;
    if (mask_y < 0 || mask_y > height - 1 || mask_x < 0 || mask_x > width - 1) {
      shared_mask[c * WARP_SIZE + pixel_id] = 0;
      continue;
    }
    const int mask_group = c / (kernel_size * kernel_size);
    const int mask_c = (2 * mask_group + 1) * kernel_size * kernel_size - c - 1;
    int mask_index =
        Loc2Index(n, mask_c, mask_y, mask_x, mask_channels, height, width);
    shared_mask[c * WARP_SIZE + pixel_id] = bottom_masks[mask_index];
  }
  __syncthreads();
  const int channels_per_group = ceilf(channels / (float)group_size);
#pragma unroll
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    int mask_group = c / channels_per_group;
    int top_index = Loc2Index(n, ph, pw, c, height, width, channels);
    scalar_t output_val = 0;
#pragma unroll
    for (int iy = start_h; iy < end_h; iy += scale_factor) {
#pragma unroll
      for (int ix = start_w; ix < end_w; ix += scale_factor) {
        if (iy < 0 || iy > height - 1 || ix < 0 || ix > width - 1) {
          continue;
        }
        int mask_iy =
            (iy - ph + (kernel_size - 1) * scale_factor / 2) / scale_factor;
        int mask_ix =
            (ix - pw + (kernel_size - 1) * scale_factor / 2) / scale_factor;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index = Loc2Index(n, iy, ix, c, height, width, channels);
        output_val +=
            shared_mask[mask_c * WARP_SIZE + pixel_id] * top_diff[feat_index];
      }
    }
    bottom_diff[top_index] = output_val;
  }
}

template <typename scalar_t>
__global__ void FeatureSum(const int num_kernels,
                           const scalar_t *__restrict__ input_data,
                           const int scale_factor, const int channels,
                           const int height, const int width,
                           scalar_t *__restrict__ output_data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    scalar_t output_val = 0;
    for (int iy = ph * scale_factor; iy < (ph + 1) * scale_factor; iy++) {
      for (int ix = pw * scale_factor; ix < (pw + 1) * scale_factor; ix++) {
        int input_id = Loc2Index(n, iy, ix, c, height * scale_factor,
                                 width * scale_factor, channels);
        output_val += input_data[input_id];
      }
    }
    const int output_id = Loc2Index(n, ph, pw, c, height, width, channels);
    output_data[output_id] = output_val;
  }
}

template <typename scalar_t>
__global__ void CARAFEBackward_Mask(const int num_kernels,
                                    const scalar_t *__restrict__ top_diff,
                                    const scalar_t *__restrict__ bottom_data,
                                    const int kernel_size, const int group_size,
                                    const int scale_factor, const int channels,
                                    const int down_height, const int down_width,
                                    const int height, const int width,
                                    const int mask_channels,
                                    scalar_t *__restrict__ mask_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }

  const int lane_id = index % WARP_SIZE;
  index = index / WARP_SIZE;
  const int mask_c = index % mask_channels;
  // (n, c, ph, pw) is an element in the bottom_data
  index = index / mask_channels;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int down_pw = pw / scale_factor;
  const int down_ph = ph / scale_factor;

  const int mask_group = mask_c / (kernel_size * kernel_size);
  const int mask_loc = mask_c % (kernel_size * kernel_size);

  const int offset_x = mask_loc % kernel_size - (kernel_size - 1) / 2;
  const int offset_y =
      mask_loc / kernel_size % kernel_size - (kernel_size - 1) / 2;

  const int down_x = down_pw + offset_x;
  const int down_y = down_ph + offset_y;

  scalar_t output_val = 0;

  if (down_y >= 0 && down_y <= down_height - 1 && down_x >= 0 &&
      down_x <= down_width - 1) {
    const int channels_per_mask = ceilf(channels / (float)group_size);
    const int start = channels_per_mask * mask_group;
    const int end = min(channels_per_mask * (mask_group + 1), channels);
    for (int c = start + lane_id; c < end; c += WARP_SIZE) {
      int bottom_id =
          Loc2Index(n, down_y, down_x, c, down_height, down_width, channels);
      int top_id = Loc2Index(n, ph, pw, c, height, width, channels);
      output_val += top_diff[top_id] * bottom_data[bottom_id];
    }
  }
  __syncwarp();
  output_val = warpReduceSum(output_val);
  if (lane_id == 0) {
    const int mask_id =
        Loc2Index(n, ph, pw, mask_c, height, width, mask_channels);
    mask_diff[mask_id] = output_val;
  }
}

int CARAFEBackwardLaucher(const at::Tensor top_grad, const at::Tensor rfeatures,
                          const at::Tensor masks, const int kernel_size,
                          const int group_size, const int scale_factor,
                          const int batch_size, const int channels,
                          const int input_height, const int input_width,
                          const int output_height, const int output_width,
                          const int mask_channels, at::Tensor rtop_grad,
                          at::Tensor rbottom_grad_hs, at::Tensor rbottom_grad,
                          at::Tensor rmask_grad, at::Tensor bottom_grad,
                          at::Tensor mask_grad) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "NCHW2NHWC_Top_Grad", ([&] {
        const scalar_t *bottom_data = top_grad.data<scalar_t>();
        scalar_t *top_data = rtop_grad.data<scalar_t>();
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, output_height * output_width, dh, dw,
                bottom_data, top_data);
      }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "CARAFELaucherBackward_Feature", ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;
        const scalar_t *top_diff = rtop_grad.data<scalar_t>();
        const scalar_t *bottom_masks = masks.data<scalar_t>();
        scalar_t *bottom_diff = rbottom_grad_hs.data<scalar_t>();

        CARAFEBackward_Feature<scalar_t>
            <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
               THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, top_diff, bottom_masks, kernel_size, group_size,
                scale_factor, channels, input_height, input_width,
                output_height, output_width, mask_channels, bottom_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "FeatureSum", ([&] {
        const int num_kernels =
            batch_size * input_height * input_width * THREADS_PER_PIXEL;
        const scalar_t *bottom_diff_hs = rbottom_grad_hs.data<scalar_t>();
        scalar_t *bottom_diff = rbottom_grad.data<scalar_t>();

        FeatureSum<scalar_t>
            <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
               THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, bottom_diff_hs, scale_factor, channels,
                input_height, input_width, bottom_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "NHWC2NCHW_Bottom_Grad", ([&] {
        const scalar_t *bottom_data = rbottom_grad.data<scalar_t>();
        scalar_t *top_data = bottom_grad.data<scalar_t>();
        const int dh = divideUP(input_height * input_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, input_height * input_width, channels, dh, dw,
                bottom_data, top_data);
      }));

  AT_DISPATCH_FLOATING_TYPES(
      top_grad.type(), "CARAFELaucherBackward_Mask", ([&] {
        const int num_kernels = batch_size * output_height * output_width *
                                mask_channels * WARP_SIZE;
        const scalar_t *top_diff = rtop_grad.data<scalar_t>();
        const scalar_t *bottom_data = rfeatures.data<scalar_t>();
        scalar_t *mask_diff = rmask_grad.data<scalar_t>();

        CARAFEBackward_Mask<scalar_t>
            <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
               THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, top_diff, bottom_data, kernel_size, group_size,
                scale_factor, channels, input_height, input_width,
                output_height, output_width, mask_channels, mask_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "NHWC2NCHW_Mask_Grad", ([&] {
        const scalar_t *bottom_data = rmask_grad.data<scalar_t>();
        scalar_t *top_data = mask_grad.data<scalar_t>();
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(mask_channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, mask_channels, dh, dw,
                bottom_data, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
