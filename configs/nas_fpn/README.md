# NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection

## Introduction

```
@inproceedings{ghiasi2019fpn,
  title={Nas-fpn: Learning scalable feature pyramid architecture for object detection},
  author={Ghiasi, Golnaz and Lin, Tsung-Yi and Le, Quoc V},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7036--7045},
  year={2019}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN. RetinaNet is used in the paper.

| Backbone    | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:-----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN    | 50e     | 12.8     | 0.513               | 15.3           | 37.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/nas_fpn/retinanet_crop640_r50_fpn_50e_190824-4d75bfa0.pth) |
| R-50-NASFPN | 50e     | 14.8     | 0.662               | 13.1           | 39.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/nas_fpn/retinanet_crop640_r50_nasfpn_50e_20191225-b82d3a86.pth) |


**Note**: We find that it is unstable to train NAS-FPN and there is a small chance that results can be 3% mAP lower.
