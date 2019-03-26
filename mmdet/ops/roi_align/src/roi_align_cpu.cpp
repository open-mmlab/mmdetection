#include <torch/torch.h>
#include <torch/extension.h>

#define CHECK_NOT_CUDA(x) AT_CHECK(!x.type().is_cuda(), #x, " must be a CPU tensor ")
#define CHECK_CONTIGUOUS(x)					\
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x)				\
  CHECK_NOT_CUDA(x);				\
  CHECK_CONTIGUOUS(x)

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
				       const int height,
				       const int width,
				       const int pooled_height,
				       const int pooled_width,
				       const int iy_upper,
				       const int ix_upper,
				       T roi_start_h,
				       T roi_start_w,
				       T bin_size_h,
				       T bin_size_w,
				       int roi_bin_grid_h,
				       int roi_bin_grid_w,
				       std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
	  static_cast<T>(iy + .5f) * bin_size_h /
	  static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
	    static_cast<T>(ix + .5f) * bin_size_w /
	    static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward_cpu_kernel(
				const int nthreads,
				const T* bottom_data,
				const T& spatial_scale,
				const int channels,
				const int height,
				const int width,
				const int pooled_height,
				const int pooled_width,
				const int sampling_ratio,
				const T* bottom_rois,
				//int roi_cols,
				T* top_data) {
  //AT_ASSERT(roi_cols == 4 || roi_cols == 5);
  int roi_cols = 5;

  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = (offset_bottom_rois[3] + 1)* spatial_scale;
    T roi_end_h = (offset_bottom_rois[4] + 1)* spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 0x0
    T roi_width = std::max(roi_end_w - roi_start_w, (T)0.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)0.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
      ? sampling_ratio
      : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
      (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
				     roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
				      height,
				      width,
				      pooled_height,
				      pooled_width,
				      roi_bin_grid_h,
				      roi_bin_grid_w,
				      roi_start_h,
				      roi_start_w,
				      bin_size_h,
				      bin_size_w,
				      roi_bin_grid_h,
				      roi_bin_grid_w,
				      pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_bottom_data =
	bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
		pc.w2 * offset_bottom_data[pc.pos2] +
		pc.w3 * offset_bottom_data[pc.pos3] +
		pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[index] = output_val;
        } // for pw
      } // for ph
    } // for c
  } // for n
}


int roi_align_forward_cpu(at::Tensor& features,
                          at::Tensor& rois,
                          int pooled_height,
                          int pooled_width,
                          float spatial_scale,
                          int sample_num,
                          at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5) {
    printf("wrong roi size....\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  const int output_size = num_rois * pooled_height * pooled_width * num_channels;

  if (output.numel() == 0) {
    return -1;
  }

  AT_DISPATCH_FLOATING_TYPES(features.type(), "ROIAlign_forward", [&] {
      ROIAlignForward_cpu_kernel<scalar_t>(
					   output_size,
					   features.data<scalar_t>(),
					   spatial_scale,
					   num_channels,
					   data_height,
					   data_width,
					   pooled_height,
					   pooled_width,
					   sample_num,
					   rois.data<scalar_t>(),
					   output.data<scalar_t>());
    });
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_align_forward_cpu, "Roi_Align forward (CPU)");
}
