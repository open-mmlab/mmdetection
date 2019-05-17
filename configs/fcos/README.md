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

| Backbone  | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN  | caffe   | 1x      | 6.9      | 0.396               | 13.6           | 36.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_fpn_1x-9f253a93.pth) |
| R-50-FPN  | caffe   | 2x      | -        | -                   | -              | 38.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_fpn_2x-f7329d80.pth) |
| R-101-FPN | caffe   | 1x      | 10.4     | 0.558               | 11.6           | 39.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r101_fpn_1x-e4889733.pth) |
| R-101-FPN | caffe   | 2x      | -        | -                   | -              | 40.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r101_fpn_2x-42e6f62d.pth) |
| X-101-64x4d-FPN | caffe   |2x | 9.7      | 0.892               | 7.0            | 42.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_x101_64x4d_fpn_2x-a36c0872.pth) |

**Notes:**
- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU for R-50 and R-101 models, and 8 GPUs with 2 image/GPU for X-101 models.
