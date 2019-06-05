# R-FCN: Object Detection via Region-based Fully Convolutional Networks

## Introduction

```
@inproceedings{dai2016r,
  title={R-fcn: Object detection via region-based fully convolutional networks},
  author={Dai, Jifeng and Li, Yi and He, Kaiming and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  pages={379--387},
  year={2016}
}
```

## Results and Models

| Backbone  | Style   | Conv      | Pool  | Lr schd | Mem (GB)  | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:| :-------: |:----: | :------:|:---------:|:-------------------:|:-----------------:|:------:|:--------:|
| R-101     | caffe   | -         | -     | 1x      | 1.9       | 0.261 | 11.2 | 77.1 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/rfcn/rfcn_r101_caffe_1x_voc0712.pth) |
| R-101     | caffe   | dconv(c5) | dpool | 1x      | 2.0       | 0.272 | 9.9  | 79.8 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/rfcn/rfcn_dconv_c5_dpool_r101_caffe_1x_voc0712.pth) |


**Notes:**
- This repo does not incorporate OHEM into RFCN.
