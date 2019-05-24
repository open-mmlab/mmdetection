# Libra R-CNN: Towards Balanced Learning for Object Detection

## Introduction

We provide config files to reproduce the results in the CVPR 2019 paper [Libra R-CNN](https://arxiv.org/pdf/1904.02701.pdf).

```
@inproceedings{pang2019libra,
  title={Libra R-CNN: Towards Balanced Learning for Object Detection},
  author={Pang, Jiangmiao and Chen, Kai and Shi, Jianping and Feng, Huajun and Ouyang, Wanli and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Architecture | Backbone  | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| Faster R-CNN | R-50-FPN        | pytorch | 1x | 4.2  | 0.375 | 12.0 | 38.6 | [model]() |
| Fast R-CNN   | R-50-FPN        | pytorch | 1x | 3.7  | 0.272 | 16.3 | 38.5 | [model]() |
| Faster R-CNN | R-101-FPN       | pytorch | 1x | 6.0  | 0.495 | 10.4 | 40.3 | [model]() |
| Faster R-CNN | X-101-64x4d-FPN | pytorch | 1x | 10.1 | 1.050 | 6.8  | 42.7 | [model]() |
| RetinaNet    | R-50-FPN        | pytorch | 1x | 3.7  | 0.328 | 11.8 | 37.7 | [model]() |
