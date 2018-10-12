// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <vector>
#include "gpu_nms.hpp"

#define CUDA_CHECK(condition)                                    \
    /* Code block avoids redefinition of cudaError_t error */    \
    do {                                                         \
        cudaError_t error = condition;                           \
        if (error != cudaSuccess) {                              \
            std::cout << cudaGetErrorString(error) << std::endl; \
        }                                                        \
    } while (0)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MULTIPLIER 16
#define LONGLONG_SIZE 64

int const threadsPerBlock =
    sizeof(unsigned long long) * 8 *
    MULTIPLIER;  // number of bits for a long long variable

__device__ inline float devIoU(float const* const a, float const* const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f),
          height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float* dev_boxes,
                           unsigned long long* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 5 + 0] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();

    unsigned long long ts[MULTIPLIER];

    if (threadIdx.x < row_size) {
#pragma unroll
        for (int i = 0; i < MULTIPLIER; ++i) {
            ts[i] = 0;
        }
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float* cur_box = dev_boxes + cur_box_idx * 5;
        int i = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
                ts[i / LONGLONG_SIZE] |= 1ULL << (i % LONGLONG_SIZE);
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);

#pragma unroll
        for (int i = 0; i < MULTIPLIER; ++i) {
            dev_mask[(cur_box_idx * col_blocks + col_start) * MULTIPLIER + i] =
                ts[i];
        }
    }
}

void _set_device(int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id) {
        return;
    }
    // The call to cudaSetDevice must come before any calls to Get, which
    // may perform initialization using the GPU.
    CUDA_CHECK(cudaSetDevice(device_id));
}

const size_t MEMORY_SIZE = 500000000;
size_t nms_Malloc() {
    float* boxes_dev = NULL;
    CUDA_CHECK(cudaMalloc(&boxes_dev, MEMORY_SIZE));
    return size_t(boxes_dev);
}

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id, size_t base) {
    _set_device(device_id);

    float* boxes_dev = NULL;
    unsigned long long* mask_dev = NULL;

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

    if (base > 0) {
        size_t require_mem =
            boxes_num * boxes_dim * sizeof(float) +
            boxes_num * col_blocks * sizeof(unsigned long long) * MULTIPLIER;
        if (require_mem >= MEMORY_SIZE) {
            std::cout << "require_mem: " << require_mem << std::endl;
        }
        boxes_dev = (float*)(base);
        mask_dev =
            (unsigned long long*)(base +
                                  512 * ((unsigned long long)(boxes_num *
                                                              boxes_dim *
                                                              sizeof(float) /
                                                              512) +
                                         1));
    } else {
        CUDA_CHECK(
            cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mask_dev, MULTIPLIER * boxes_num * col_blocks *
                                             sizeof(unsigned long long)));
    }
    CUDA_CHECK(cudaMemcpy(boxes_dev, boxes_host,
                          boxes_num * boxes_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev,
                                    mask_dev);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks *
                                              MULTIPLIER);
    CUDA_CHECK(cudaMemcpy(
        &mask_host[0], mask_dev,
        sizeof(unsigned long long) * boxes_num * col_blocks * MULTIPLIER,
        cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks * MULTIPLIER);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks * MULTIPLIER);

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;
        int offset = inblock / LONGLONG_SIZE;
        int bit_pos = inblock % LONGLONG_SIZE;

        if (!(remv[nblock * MULTIPLIER + offset] & (1ULL << bit_pos))) {
            keep_out[num_to_keep++] = i;
            unsigned long long* p = &mask_host[0] + i * col_blocks * MULTIPLIER;
            for (int j = nblock * MULTIPLIER + offset;
                 j < col_blocks * MULTIPLIER; j++) {
                remv[j] |= p[j];
            }
        }
    }
    *num_out = num_to_keep;

    if (!base) {
        CUDA_CHECK(cudaFree(boxes_dev));
        CUDA_CHECK(cudaFree(mask_dev));
    }
}
