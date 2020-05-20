# Grid R-CNN

## Introduction

```
@inproceedings{lu2019grid,
  title={Grid r-cnn},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{lu2019grid,
  title={Grid R-CNN Plus: Faster and Better},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  journal={arXiv preprint arXiv:1906.05688},
  year={2019}
}
```

## Results and Models

| Backbone    | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:-----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50        | 2x      | 4.8      | 1.172               | 10.9           | 40.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/grid_rcnn/grid_rcnn_gn_head_r50_fpn_2x_20190619-5b29cf9d.pth) |
| R-101       | 2x      | 6.7      | 1.214               | 10.0           | 41.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/grid_rcnn/grid_rcnn_gn_head_r101_fpn_2x_20190619-a4b61645.pth) |
| X-101-32x4d | 2x      | 8.0      | 1.335               | 8.5            | 43.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/grid_rcnn/grid_rcnn_gn_head_x101_32x4d_fpn_2x_20190619-0bbfd87a.pth) |
| X-101-64x4d | 2x      | 10.9     | 1.753               | 6.4            | 43.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/grid_rcnn/grid_rcnn_gn_head_x101_64x4d_fpn_2x_20190619-8f4e20bb.pth) |

**Notes:**
- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
- The warming up lasts for 1 epoch and `2x` here indicates 25 epochs.
