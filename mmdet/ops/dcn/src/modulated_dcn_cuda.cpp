#include <torch/torch.h>

#include <cmath>
#include <vector>

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


void modulated_deformable_im2col_cuda(const at::Tensor data_im, const at::Tensor data_offset,
                                      const at::Tensor data_mask, const int batch_size, const int channels,
                                      const int height_im, const int width_im, const int height_col,
                                      const int width_col, const int kernel_h, const int kenerl_w,
                                      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                      const int dilation_h, const int dilation_w,
                                      const int deformable_group, at::Tensor data_col);

void modulated_deformable_col2im_cuda(const at::Tensor data_col, const at::Tensor data_offset,
                                      const at::Tensor data_mask, const int batch_size, const int channels,
                                      const int height_im, const int width_im, const int height_col,
                                      const int width_col, const int kernel_h, const int kenerl_w,
                                      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                      const int dilation_h, const int dilation_w,
                                      const int deformable_group, at::Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(const at::Tensor data_col, const at::Tensor data_im,
                                            const at::Tensor data_offset, const at::Tensor data_mask,
                                            const int batch_size, const int channels, const int height_im,
                                            const int width_im, const int height_col, const int width_col,
                                            const int kernel_h, const int kenerl_w, const int pad_h,
                                            const int pad_w, const int stride_h, const int stride_w,
                                            const int dilation_h, const int dilation_w,
                                            const int deformable_group, at::Tensor grad_offset,
                                            at::Tensor grad_mask);


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
                                const float trans_std);

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
                                    const float trans_std);


void modulated_deform_conv_cuda_forward(at::Tensor input, at::Tensor weight,
                         at::Tensor bias, at::Tensor ones,
                         at::Tensor offset, at::Tensor mask,
                         at::Tensor output, at::Tensor columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 8, input, weight, bias, ones, offset, mask, output, columns));
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
        channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 ||
        ones.size(0) * ones.size(1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.type());
    }

    // resize output
    output = output.view({batch, channels_out, height_out, width_out});
    // resize temporary columns
    columns = at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.type());

    for (int b = 0; b < batch; b++)
    {
        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        output[b] = output[b].flatten(1).addmm_(bias.view({-1, 1}), ones.view({1, -1}), 0.0f, 1.0f).view_as(output[b]);

        modulated_deformable_im2col_cuda(input[b], offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, columns);

        //(k * m)  x  (m * n)
        // Y = WC
        output[b] = output[b].flatten(1).addmm_(weight.flatten(1), columns).view_as(output[b]);
    }
}

void modulated_deform_conv_cuda_backward(at::Tensor input, at::Tensor weight,
                          at::Tensor bias, at::Tensor ones,
                          at::Tensor offset, at::Tensor mask,
                          at::Tensor columns,
                          at::Tensor grad_input, at::Tensor grad_weight,
                          at::Tensor grad_bias, at::Tensor grad_offset,
                          at::Tensor grad_mask, at::Tensor grad_output,
                          int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 13, input, weight, bias, ones, offset, mask, columns,
    //                                        grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output));
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
        channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 ||
        ones.size(0) * ones.size(1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.type());
    }

    grad_input = grad_input.view({batch, channels, height, width});
    columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out}, input.type());

    for (int b = 0; b < batch; b++)
    {
        columns.addmm_(weight.flatten(1).transpose(0, 1), grad_output[b].flatten(1), 0.0f, 1.0f);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(columns, input[b], offset[b], mask[b],
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset[b], grad_mask[b]);
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(columns, offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input[b]);

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(input[b], offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns);

        grad_weight = grad_weight.flatten(1).addmm_(grad_output[b].flatten(1), columns.transpose(0, 1)).view_as(grad_weight);

        grad_bias = grad_bias.view({-1, 1}).addmm_(grad_output[b].flatten(1), ones.view({-1, 1})).view(-1);
    }

}

void deform_psroi_pooling_cuda_forward(at::Tensor  input, at::Tensor  bbox,
                                       at::Tensor  trans,
                                       at::Tensor  out, at::Tensor  top_count,
                                       const int no_trans,
                                       const float spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const float trans_std)
{
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, bbox, trans, out, top_count));

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int channels_trans = no_trans? 2 : trans.size(1);

    const int num_bbox = bbox.size(0);
    if (num_bbox != out.size(0))
        AT_ERROR("Output shape and bbox number wont match: (%d vs %d).",
                out.size(0), num_bbox);

    DeformablePSROIPoolForward(input, bbox, trans, out, top_count,
                               batch, channels, height, width,
                               num_bbox,
                               channels_trans,
                               no_trans,
                               spatial_scale,
                               output_dim,
                               group_size,
                               pooled_size,
                               part_size,
                               sample_per_part,
                               trans_std);
}

void deform_psroi_pooling_cuda_backward(at::Tensor  out_grad,
                                        at::Tensor  input, at::Tensor  bbox,
                                        at::Tensor  trans, at::Tensor  top_count,
                                        at::Tensor  input_grad, at::Tensor  trans_grad,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std)
{
    AT_CHECK(out_grad.is_contiguous(), "out_grad tensor has to be contiguous");
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, bbox, trans, out_grad, top_count,
    //                 input_grad, trans_grad));

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int channels_trans = no_trans? 2 : trans.size(1);

    const int num_bbox = bbox.size(0);
    if (num_bbox != out_grad.size(0))
        AT_ERROR("Output shape and bbox number wont match: (%d vs %d).",
                out_grad.size(0), num_bbox);

    DeformablePSROIPoolBackwardAcc(out_grad,
                                   input,
                                   bbox,
                                   trans,
                                   top_count,
                                   input_grad,
                                   trans_grad,
                                   batch, channels, height, width, num_bbox,
                                   channels_trans,
                                   no_trans,
                                   spatial_scale,
                                   output_dim,
                                   group_size,
                                   pooled_size,
                                   part_size,
                                   sample_per_part,
                                   trans_std);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("modulated_deform_conv_cuda_forward", &modulated_deform_conv_cuda_forward,
          "modulated deform conv forward (CUDA)");
    m.def("modulated_deform_conv_cuda_backward", &modulated_deform_conv_cuda_backward,
          "modulated deform conv backward (CUDA)");
    m.def("deform_psroi_pooling_cuda_forward", &deform_psroi_pooling_cuda_forward,
          "deform psroi pooling forward(CUDA)");
    m.def("deform_psroi_pooling_cuda_backward", &deform_psroi_pooling_cuda_backward,
          "deform psroi pooling backward(CUDA)");
}