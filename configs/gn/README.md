# Group Normalization

## Introduction

```
@inproceedings{wu2018group,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## Results and Models

| Backbone      | model      | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:-------------:|:----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.2      | 0.806               | 5.4            | 39.8   | 36.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth) |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.2      | 0.806               | 5.4            | 40.1   | 36.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r50_fpn_gn_3x_20180113-8e82f48d.pth) |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 0.970               | 4.8            | 41.5   | 37.0    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r101_fpn_gn_2x_20180113-9598649c.pth) |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      | 0.970               | 4.8            | 41.6   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r101_fpn_gn_3x_20180113-a14ffb96.pth) |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.2      | 0.806               | 5.4            | 39.7   | 35.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r50_fpn_gn_contrib_2x_20180113-ec93305c.pth) |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.2      | 0.806               | 5.4            | 40.0   | 36.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/gn/mask_rcnn_r50_fpn_gn_contrib_3x_20180113-9d230cab.pth) |

**Notes:**
- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- **Memory, Train/Inf time is outdated.**