# Mask Scoring R-CNN

## Introduction

```
@article{huang2019mask,
  title={Mask Scoring R-CNN},
  author={Huang, Zhaojin and Huang, Lichao and Gong, Yongchao and Huang, Chang and Wang, Xinggang},
  journal={arXiv preprint arXiv:1903.00241},
  year={2019}
}
```

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:-------------:|:----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN      | caffe      | 1x      | 4.3      | 0.537               | 10.1           | 37.5   | 35.6    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r50_caffe_fpn_1x-234dfcbd.pth) |
| R-101-FPN     | caffe      | 1x      | 6.2      | 0.682               |  9.1           | 40.0   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r101_caffe_fpn_1x-3aac0304.pth) |
