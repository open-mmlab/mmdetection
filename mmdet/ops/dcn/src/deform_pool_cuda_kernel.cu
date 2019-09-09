/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/
// modify from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/cuda/deform_psroi_pooling_cuda.cu

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t bilinear_interp(
    const scalar_t *data,
    const scalar_t x,
    const scalar_t y,
    const int width,
    const int height)
{
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  scalar_t dist_x = (scalar_t)(x - x1);
  scalar_t dist_y = (scalar_t)(y - y1);
  scalar_t value11 = data[y1 * width + x1];
  scalar_t value12 = data[y2 * width + x1];
  scalar_t value21 = data[y1 * width + x2];
  scalar_t value22 = data[y2 * width + x2];
  scalar_t value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12 + dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
  return value;
}

template <typename scalar_t>
__global__ void DeformablePSROIPoolForwardKernel(
    const int count,
    const scalar_t *bottom_data,
    const scalar_t spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const scalar_t *bottom_rois, const scalar_t *bottom_trans,
    const int no_trans,
    const scalar_t trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    scalar_t *top_data,
    scalar_t *top_count)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = (scalar_t)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    scalar_t roi_start_h = (scalar_t)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    scalar_t roi_end_w = (scalar_t)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    scalar_t roi_end_h = (scalar_t)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / (scalar_t)(pooled_height);
    scalar_t bin_size_w = roi_width / (scalar_t)(pooled_width);

    scalar_t sub_bin_size_h = bin_size_h / (scalar_t)(sample_per_part);
    scalar_t sub_bin_size_w = bin_size_w / (scalar_t)(sample_per_part);

    int part_h = floor((scalar_t)(ph) / pooled_height * part_size);
    int part_w = floor((scalar_t)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    scalar_t trans_x = no_trans ? (scalar_t)(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * (scalar_t)trans_std;
    scalar_t trans_y = no_trans ? (scalar_t)(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * (scalar_t)trans_std;

    scalar_t wstart = (scalar_t)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    scalar_t hstart = (scalar_t)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    scalar_t sum = 0;
    int count = 0;
    int gw = floor((scalar_t)(pw)*group_size / pooled_width);
    int gh = floor((scalar_t)(ph)*group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        scalar_t w = wstart + iw * sub_bin_size_w;
        scalar_t h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        scalar_t val = bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        count++;
      }
    }
    top_data[index] = count == 0 ? (scalar_t)(0) : sum / count;
    top_count[index] = count;
  }
}

template <typename scalar_t>
__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count,
    const scalar_t *top_diff,
    const scalar_t *top_count,
    const int num_rois,
    const scalar_t spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    scalar_t *bottom_data_diff, scalar_t *bottom_trans_diff,
    const scalar_t *bottom_data,
    const scalar_t *bottom_rois,
    const scalar_t *bottom_trans,
    const int no_trans,
    const scalar_t trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = (scalar_t)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    scalar_t roi_start_h = (scalar_t)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    scalar_t roi_end_w = (scalar_t)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    scalar_t roi_end_h = (scalar_t)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / (scalar_t)(pooled_height);
    scalar_t bin_size_w = roi_width / (scalar_t)(pooled_width);

    scalar_t sub_bin_size_h = bin_size_h / (scalar_t)(sample_per_part);
    scalar_t sub_bin_size_w = bin_size_w / (scalar_t)(sample_per_part);

    int part_h = floor((scalar_t)(ph) / pooled_height * part_size);
    int part_w = floor((scalar_t)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    scalar_t trans_x = no_trans ? (scalar_t)(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * (scalar_t)trans_std;
    scalar_t trans_y = no_trans ? (scalar_t)(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * (scalar_t)trans_std;

    scalar_t wstart = (scalar_t)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    scalar_t hstart = (scalar_t)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0)
    {
      continue;
    }
    scalar_t diff_val = top_diff[index] / top_count[index];
    const scalar_t *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
    scalar_t *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = floor((scalar_t)(pw)*group_size / pooled_width);
    int gh = floor((scalar_t)(ph)*group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        scalar_t w = wstart + iw * sub_bin_size_w;
        scalar_t h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        // backward on feature
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);
        scalar_t dist_x = w - x0, dist_y = h - y0;
        scalar_t q00 = (1 - dist_x) * (1 - dist_y);
        scalar_t q01 = (1 - dist_x) * dist_y;
        scalar_t q10 = dist_x * (1 - dist_y);
        scalar_t q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);

        if (no_trans)
        {
          continue;
        }
        scalar_t U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        scalar_t U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        scalar_t U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        scalar_t U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
        scalar_t diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;
        diff_x *= roi_width;
        scalar_t diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;
        diff_y *= roi_height;

        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);
      }
    }
  }
}

void DeformablePSROIPoolForward(const at::Tensor data,
                                const at::Tensor bbox,
                                const at::Tensor trans,
                                at::Tensor out,
                                at::Tensor top_count,
                                const int batch,
                                const int channels,
                                const int height,
                                const int width,
                                const int num_bbox,
                                const int channels_trans,
                                const int no_trans,
                                const float spatial_scale,
                                const int output_dim,
                                const int group_size,
                                const int pooled_size,
                                const int part_size,
                                const int sample_per_part,
                                const float trans_std)
{
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.scalar_type(), "deformable_psroi_pool_forward", ([&] {
        const scalar_t *bottom_data = data.data<scalar_t>();
        const scalar_t *bottom_rois = bbox.data<scalar_t>();
        const scalar_t *bottom_trans = no_trans ? NULL : trans.data<scalar_t>();
        scalar_t *top_data = out.data<scalar_t>();
        scalar_t *top_count_data = top_count.data<scalar_t>();

        DeformablePSROIPoolForwardKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, bottom_data, (scalar_t)spatial_scale, channels, height, width, pooled_height, pooled_width,
            bottom_rois, bottom_trans, no_trans, (scalar_t)trans_std, sample_per_part, output_dim,
            group_size, part_size, num_classes, channels_each_class, top_data, top_count_data);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}

void DeformablePSROIPoolBackwardAcc(const at::Tensor out_grad,
                                    const at::Tensor data,
                                    const at::Tensor bbox,
                                    const at::Tensor trans,
                                    const at::Tensor top_count,
                                    at::Tensor in_grad,
                                    at::Tensor trans_grad,
                                    const int batch,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int num_bbox,
                                    const int channels_trans,
                                    const int no_trans,
                                    const float spatial_scale,
                                    const int output_dim,
                                    const int group_size,
                                    const int pooled_size,
                                    const int part_size,
                                    const int sample_per_part,
                                    const float trans_std)
{
  // LOG(INFO) << "DeformablePSROIPoolBackward";
  const int num_rois = num_bbox;
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.scalar_type(), "deformable_psroi_pool_backward_acc", ([&] {
        const scalar_t *top_diff = out_grad.data<scalar_t>();
        const scalar_t *bottom_data = data.data<scalar_t>();
        const scalar_t *bottom_rois = bbox.data<scalar_t>();
        const scalar_t *bottom_trans = no_trans ? NULL : trans.data<scalar_t>();
        scalar_t *bottom_data_diff = in_grad.data<scalar_t>();
        scalar_t *bottom_trans_diff = no_trans ? NULL : trans_grad.data<scalar_t>();
        const scalar_t *top_count_data = top_count.data<scalar_t>();

        DeformablePSROIPoolBackwardAccKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, top_diff, top_count_data, num_rois, (scalar_t)spatial_scale, channels, height, width,
            pooled_height, pooled_width, output_dim, bottom_data_diff, bottom_trans_diff,
            bottom_data, bottom_rois, bottom_trans, no_trans, (scalar_t)trans_std, sample_per_part,
            group_size, part_size, num_classes, channels_each_class);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}