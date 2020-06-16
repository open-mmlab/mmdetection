# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection


## Introduction

We provide config files to reproduce the object detection results in the paper [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

```
@article{GFocaL,
  title   = {Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author  = {Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang},
  journal = {arXiv preprint arXiv:2006.04388},
  year    = {2020}
}
```


## Results and Models

| Backbone          | Style   | Lr schd | Multi-scale Training| Inf time (fps) | box AP | Download |
|:-----------------:|:-------:|:-------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50              | pytorch | 1x      | No                  | 19.5           |    | [model]() &#124; [log]() |
| R-50              | pytorch | 2x      | Yes                 | 19.5           |    | [model]() &#124; [log]() |
| R-101             | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |
| R-101-dcnv2       | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |
| X-101-32x4d       | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |
| X-101-32x4d-dcnv2 | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |
| X-101-64x4d       | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |
| X-101-64x4d-dcnv2 | pytorch | 2x      | Yes                 |            |    | [model]() &#124; [log]() |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2.* \
[5] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*
