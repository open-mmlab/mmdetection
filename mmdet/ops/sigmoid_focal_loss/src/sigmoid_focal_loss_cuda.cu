// modified from
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/cuda/SigmoidFocalLoss_cuda.cu

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// This file is modified from
// https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu
// Cheng-Yang Fu
// cyfu@cs.unc.edu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void SigmoidFocalLossForward(const int nthreads,
                                        const scalar_t *logits,
                                        const int64_t *targets,
                                        const int num_classes,
                                        const float gamma, const float alpha,
                                        const int num, scalar_t *losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int n = i / num_classes;
    int d = i % num_classes;  // current class[0~79];
    int t = targets[n];       // target class [1~80];

    // Decide it is positive or negative case.
    scalar_t c1 = (t == (d + 1));
    scalar_t c2 = (t >= 0 & t != (d + 1));

    scalar_t zn = (1.0 - alpha);
    scalar_t zp = (alpha);

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    scalar_t p = 1. / (1. + expf(-logits[i]));

    // (1-p)**gamma * log(p) where
    scalar_t term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));

    // p**gamma * log(1-p)
    scalar_t term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;

  }  // CUDA_1D_KERNEL_LOOP
}  // SigmoidFocalLossForward

template <typename scalar_t>
__global__ void SigmoidFocalLossBackward(
    const int nthreads, const scalar_t *logits, const int64_t *targets,
    const scalar_t *d_losses, const int num_classes, const float gamma,
    const float alpha, const int num, scalar_t *d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int n = i / num_classes;
    int d = i % num_classes;  // current class[0~79];
    int t = targets[n];       // target class [1~80], 0 is background;

    // Decide it is positive or negative case.
    scalar_t c1 = (t == (d + 1));
    scalar_t c2 = (t >= 0 & t != (d + 1));

    scalar_t zn = (1.0 - alpha);
    scalar_t zp = (alpha);
    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    scalar_t p = 1. / (1. + expf(-logits[i]));

    // (1-p)**g * (1 - p - g*p*log(p)
    scalar_t term1 =
        powf((1. - p), gamma) * (1. - p - (p * gamma * logf(max(p, FLT_MIN))));

    // (p**g) * (g*(1-p)*log(1-p) - p)
    scalar_t term2 =
        powf(p, gamma) *
        ((-1. * logits[i] * (logits[i] >= 0) -
          logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
             (1. - p) * gamma -
         p);
    d_logits[i] = 0.0;
    d_logits[i] += -c1 * term1 * zp;
    d_logits[i] += -c2 * term2 * zn;
    d_logits[i] = d_logits[i] * d_losses[i];

  }  // CUDA_1D_KERNEL_LOOP
}  // SigmoidFocalLossBackward

at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                         const at::Tensor &targets,
                                         const int num_classes,
                                         const float gamma, const float alpha) {
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);

  auto losses = at::empty({num_samples, logits.size(1)}, logits.options());
  auto losses_size = num_samples * logits.size(1);

  dim3 grid(
      std::min(THCCeilDiv((int64_t)losses_size, (int64_t)512), (int64_t)4096));
  dim3 block(512);

  if (losses.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
        SigmoidFocalLossForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            losses_size, logits.contiguous().data<scalar_t>(),
            targets.contiguous().data<int64_t>(), num_classes, gamma, alpha,
            num_samples, losses.data<scalar_t>());
      });
  THCudaCheck(cudaGetLastError());
  return losses;
}

at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                          const at::Tensor &targets,
                                          const at::Tensor &d_losses,
                                          const int num_classes,
                                          const float gamma,
                                          const float alpha) {
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.type().is_cuda(), "d_losses must be a CUDA tensor");

  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
  AT_ASSERTM(logits.size(1) == num_classes,
             "logits.size(1) should be num_classes");

  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  auto d_logits_size = num_samples * logits.size(1);

  dim3 grid(std::min(THCCeilDiv((int64_t)d_logits_size, (int64_t)512),
                     (int64_t)4096));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
        SigmoidFocalLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            d_logits_size, logits.contiguous().data<scalar_t>(),
            targets.contiguous().data<int64_t>(),
            d_losses.contiguous().data<scalar_t>(), num_classes, gamma, alpha,
            num_samples, d_logits.data<scalar_t>());
      });

  THCudaCheck(cudaGetLastError());
  return d_logits;
}
