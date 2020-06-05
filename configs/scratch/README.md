# Rethinking ImageNet Pre-training

## Introduction

```
@article{he2018rethinking,
  title={Rethinking imagenet pre-training},
  author={He, Kaiming and Girshick, Ross and Doll{\'a}r, Piotr},
  journal={arXiv preprint arXiv:1811.08883},
  year={2018}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | box AP | mask AP | Download |
|:------------:|:---------:|:-------:|:-------:|:------:|:-------:|:--------:|
| Faster R-CNN | R-50-FPN  | pytorch | 6x      |        |         |          |
| Mask R-CNN   | R-50-FPN  | pytorch | 6x      |        |         |          |

Note:
- The above models are trained with 16 GPUs.
