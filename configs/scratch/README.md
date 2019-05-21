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
| Faster R-CNN | R-50-FPN  | pytorch | 6x      | 40.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/scratch/scratch_faster_rcnn_r50_fpn_gn_6x-20190515-ff554978.pth) |
| Mask R-CNN   | R-50-FPN  | pytorch | 6x      | 41.0   | 37.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/scratch/scratch_mask_rcnn_r50_fpn_gn_6x_20190515-96743f5e.pth) |

Note:
- The above models are trained with 16 GPUs.