#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <vector>

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
__global__ void PSROIPoolForward(
    const int nthreads, const scalar_t *bottom_data, const scalar_t *rois,
    const scalar_t spatial_scale, const int channels, const int height,
    const int width, const int pooled_h, const int pooled_w,
    const int group_size, const int out_chn, scalar_t *top_data,
    int *mapping_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, ctop, ph, pw) is an element in the pooled output
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int ctop = (index / pooled_w / pooled_h) % out_chn;
    int n = index / pooled_w / pooled_h / out_chn;

    const scalar_t *offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    // calculate the roi region on feature maps
    scalar_t roi_x1 =
        static_cast<scalar_t>(round(offset_rois[1])) * spatial_scale;
    scalar_t roi_y1 =
        static_cast<scalar_t>(round(offset_rois[2])) * spatial_scale;
    scalar_t roi_x2 =
        static_cast<scalar_t>(round(offset_rois[3]) + 1) * spatial_scale;
    scalar_t roi_y2 =
        static_cast<scalar_t>(round(offset_rois[4]) + 1) * spatial_scale;

    // force malformed rois to be 1x1
    scalar_t roi_w = max(roi_x2 - roi_x1, 0.1);
    scalar_t roi_h = max(roi_y2 - roi_y1, 0.1);

    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_w) / static_cast<scalar_t>(pooled_w);
    scalar_t bin_size_h =
        static_cast<scalar_t>(roi_h) / static_cast<scalar_t>(pooled_h);

    // the corresponding bin region
    int bin_x1 = floor(static_cast<scalar_t>(pw) * bin_size_w + roi_x1);
    int bin_y1 = floor(static_cast<scalar_t>(ph) * bin_size_h + roi_y1);
    int bin_x2 = ceil(static_cast<scalar_t>(pw + 1) * bin_size_w + roi_x1);
    int bin_y2 = ceil(static_cast<scalar_t>(ph + 1) * bin_size_h + roi_y1);

    // add roi offsets and clip to input boundaries
    bin_x1 = min(max(bin_x1, 0), width);
    bin_y1 = min(max(bin_y1, 0), height);
    bin_x2 = min(max(bin_x2, 0), width);
    bin_y2 = min(max(bin_y2, 0), height);
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    // the corresponding input channel
    int gw = floor(static_cast<scalar_t>(pw) * group_size / pooled_w);
    int gh = floor(static_cast<scalar_t>(ph) * group_size / pooled_h);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop * group_size + gh) * group_size + gw;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    scalar_t out_sum = 0;
    for (int h = bin_y1; h < bin_y2; ++h) {
      for (int w = bin_x1; w < bin_x2; ++w) {
        int offset = h * width + w;
        out_sum += bottom_data[offset];
      }
    }
    scalar_t bin_area = (bin_y2 - bin_y1) * (bin_x2 - bin_x1);
    top_data[index] = is_empty ? 0. : out_sum / bin_area;
    mapping_channel[index] = c;
  }
}

int PSROIPoolForwardLauncher(const at::Tensor features, const at::Tensor rois,
                             const float spatial_scale, const int channels,
                             const int height, const int width,
                             const int num_rois, const int pooled_h,
                             const int pooled_w, const int group_size,
                             const int out_chn, at::Tensor output,
                             at::Tensor mapping_channel) {
  const int output_size = num_rois * out_chn * pooled_h * pooled_w;

  AT_DISPATCH_FLOATING_TYPES(
      features.type(), "PSROIPoolLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        int *mapping_channel_data = mapping_channel.data<int>();

        PSROIPoolForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                channels, height, width, pooled_h, pooled_w, group_size,
                out_chn, top_data, mapping_channel_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void PSROIPoolBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *rois,
    const int *mapping_channel, const scalar_t spatial_scale,
    const int channels, const int height, const int width, const int pooled_h,
    const int pooled_w, const int out_chn, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int n = index / pooled_w / pooled_h / out_chn;

    const scalar_t *offset_rois = rois + n * 5;
    int roi_batch_ind = rois[0];
    scalar_t roi_x1 =
        static_cast<scalar_t>(round(offset_rois[1])) * spatial_scale;
    scalar_t roi_y1 =
        static_cast<scalar_t>(round(offset_rois[2])) * spatial_scale;
    scalar_t roi_x2 =
        static_cast<scalar_t>(round(offset_rois[3]) + 1) * spatial_scale;
    scalar_t roi_y2 =
        static_cast<scalar_t>(round(offset_rois[4]) + 1) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_w = max(roi_x2 - roi_x1, 0.1);
    scalar_t roi_h = max(roi_y2 - roi_y1, 0.1);

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_h / static_cast<scalar_t>(pooled_h);
    scalar_t bin_size_w = roi_w / static_cast<scalar_t>(pooled_w);

    int bin_x1 = floor(static_cast<scalar_t>(pw) * bin_size_w + roi_x1);
    int bin_y1 = floor(static_cast<scalar_t>(ph) * bin_size_h + roi_y1);
    int bin_x2 = ceil(static_cast<scalar_t>(pw + 1) * bin_size_w + roi_x1);
    int bin_y2 = ceil(static_cast<scalar_t>(ph + 1) * bin_size_h + roi_y1);
    // add roi offsets and clip to input boundaries
    bin_y1 = min(max(bin_y1, 0), height);
    bin_y2 = min(max(bin_y2, 0), height);
    bin_x1 = min(max(bin_x1, 0), width);
    bin_x2 = min(max(bin_x2, 0), width);
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    // Compute c at bottom
    int c = mapping_channel[index];
    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    scalar_t bin_area = (bin_y2 - bin_y1) * (bin_x2 - bin_x1);
    scalar_t diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    for (int h = bin_y1; h < bin_y2; ++h) {
      for (int w = bin_x1; w < bin_x2; ++w) {
        int bottom_index = h * width + w;
        atomicAdd(offset_bottom_diff + bottom_index, diff_val);
      }
    }
  }
}

template <>
__global__ void PSROIPoolBackward<double>(
    const int nthreads, const double *top_diff, const double *rois,
    const int *mapping_channel, const double spatial_scale, const int channels,
    const int height, const int width, const int pooled_h, const int pooled_w,
    const int out_chn, double *bottom_diff) {}

int PSROIPoolBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois,
                              const at::Tensor mapping_channel,
                              const float spatial_scale, const int channels,
                              const int height, const int width,
                              const int num_rois, const int pooled_h,
                              const int pooled_w, const int out_chn,
                              at::Tensor bottom_grad) {
  const int output_size = num_rois * out_chn * pooled_h * pooled_w;
  AT_DISPATCH_FLOATING_TYPES(
      top_grad.type(), "PSROIPoolLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        const int *mapping_channel_data = mapping_channel.data<int>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();

        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        PSROIPoolBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, mapping_channel_data,
                scalar_t(spatial_scale), channels, height, width, pooled_h,
                pooled_w, out_chn, bottom_diff);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
