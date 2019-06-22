# FCOS: Fully Convolutional One-Stage Object Detection

## Introduction

```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```

## Results and Models

| Backbone  | Style   | GN  | MS train | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | N       | N       | 1x      | 5.5      | 0.373               | 13.7           | 35.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth) |
| R-50      | caffe   | Y       | N       | 1x      | 6.9      | 0.396               | 13.6           | 36.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_20190516-9f253a93.pth) |
| R-50      | caffe   | Y       | N       | 2x      | -        | -                   | -              | 36.9   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_gn_2x_4gpu_20190516_-93484354.pth) |
| R-101     | caffe   | Y       | N       | 1x      | 10.4     | 0.558               | 11.6           | 39.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r101_caffe_fpn_gn_1x_4gpu_20190516-e4889733.pth) |
| R-101     | caffe   | Y       | N       | 2x      | -        | -                   | -              | 39.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r101_caffe_fpn_gn_2x_4gpu_20190516-c03af97b.pth) |


| Backbone  | Style   | GN  | MS train | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | Y       | Y       | 2x      | -        | -                   | -              | 38.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth) |
| R-101     | caffe   | Y       | Y       | 2x      | -        | -                   | -              | 40.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20190516-42e6f62d.pth) |
| X-101     | caffe   | Y       | Y       | 2x      | 9.7      | 0.892               | 7.0            | 42.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_20190516-a36c0872.pth) |

**Notes:**
- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU for R-50 and R-101 models, and 8 GPUs with 2 image/GPU for X-101 models.
- The X-101 backbone is X-101-64x4d.
