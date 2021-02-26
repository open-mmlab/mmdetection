# Feature Pyramid Grids

## Introduction

```latex
@article{chen2020feature,
  title={Feature pyramid grids},
  author={Chen, Kai and Cao, Yuhang and Loy, Chen Change and Lin, Dahua and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2004.03580},
  year={2020}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN.
All backbones are Resnet-50 in pytorch style.

| Method       | Neck        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:-----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:-------:|:--------:|
| Faster R-CNN | FPG         | 50e     | 20.0     | -              | 42.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg_crop640_50e_coco.py) |
| Faster R-CNN | FPG-chn128  | 50e     | 11.9     | -              | 41.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |
| Mask R-CNN   | FPG         | 50e     | 23.2     | -              | 42.7   | 37.8    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg_crop640_50e_coco.py) |
| Mask R-CNN   | FPG-chn128  | 50e     | 15.3     | -              | 41.7   | 36.9    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |
| RetinaNet    | FPG         | 50e     | 20.8     | -              | 40.5   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py) |
| RetinaNet    | FPG-chn128  | 50e     |          | -              |        | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco.py) |

**Note**: Chn128 means to decrease the number of channels of features and convs from 256 (default) to 128 in
Neck and BBox Head, which can greatly decrease memory consumption without sacrificing much precision.
