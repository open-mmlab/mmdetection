# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection


## Introduction

We provide config files to reproduce the object detection results in the paper [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

```
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```


## Results and Models

| Backbone          | Style   | Lr schd | Multi-scale Training| Inf time (fps) | box AP | Download |
|:-----------------:|:-------:|:-------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50              | pytorch | 1x      | No                  | 19.5           | 40.2   | [model](https://drive.google.com/file/d/1lznguKfDocte6Ur-7wc1V31QxQZm4OQs/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1Wyia0lsSVNzomUlvtu95Um_GpOfwucN4/view?usp=sharing) |
| R-50              | pytorch | 2x      | Yes                 | 19.5           | 42.9   | [model](https://drive.google.com/file/d/1RN19ndpKlnFGazor-C6NvOsyUlJVQIPI/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1U_XPe61qaYIn_3n-VM-1JTB_LM8NwNA9/view?usp=sharing) |
| R-101             | pytorch | 2x      | Yes                 | 14.7           | 44.7   | [model](https://drive.google.com/file/d/1WKFcvv1kerYdMuSMVcRezRTk0FH5a6LK/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1sFnxPUPHM_PohelvCzkJJfQDnfkzmDqg/view?usp=sharing) |
| R-101-dcnv2       | pytorch | 2x      | Yes                 | 12.9           | 47.1   | [model](https://drive.google.com/file/d/1Fp-nLJYPBsohI5JPWOEw9383oxbbxXXe/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/13aiU_gFevQQaDapo8bg7rxi3qU-e4YLl/view?usp=sharing) |
| X-101-32x4d       | pytorch | 2x      | Yes                 | 12.1           | 45.9   | [model](https://drive.google.com/file/d/1LTVw8GSMbCGB6wDjqkou934Yl32pVGac/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/10FsArE_cJHFhUtn7Og0Z-ZwzKZdwyeh_/view?usp=sharing) |
| X-101-32x4d-dcnv2 | pytorch | 2x      | Yes                 | 10.7           | 48.2   | [model](https://drive.google.com/file/d/1ULjoJ8H71phrkFOKH4uCzqn9WZnGHAsd/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/12JysUE3pBuIXSaprupFfRlE9_fQY6Mez/view?usp=sharing) |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2.* \
[4] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*
