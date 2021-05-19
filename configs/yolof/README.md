# You Only Look One-level Feature

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{chen2021you,
  title={You Only Look One-level Feature},
  author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Results and Models

| Backbone  | Style   | Epoch | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50-C5     | caffe | Y | 1x      | 8.3      |   37.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py)       |[model](http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427.log.json) |

**Note**:

1. We find that the performance is unstable and may fluctuate by about 0.3 mAP. mAP 37.4 ~ 37.7 is acceptable in YOLOF_R_50_C5_1x. Such fluctuation can also be found in the [original implementation](https://github.com/chensnathan/YOLOF).
2. In addition to instability issues, sometimes there are large loss fluctuations and NAN, so there may still be problems with this project, which will be improved subsequently.
