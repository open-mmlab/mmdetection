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

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50        | 2x      | 5.1      | 15.0           | 40.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130_221140.log.json) |
| R-101       | 2x      | 7.0      | 12.6           | 41.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309_164224.log.json) |
| X-101-32x4d | 2x      | 8.3      | 10.8           | 42.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130_215413.log.json) |
| X-101-64x4d | 2x      | 11.3     | 7.7            | 43.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204_080641.log.json) |

**Notes:**
- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
- The warming up lasts for 1 epoch and `2x` here indicates 25 epochs.
