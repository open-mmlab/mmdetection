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
| R-50-FPN    |         |          |                     |                |        |          |
| R-50-NASFPN |         |          |                     |                |        |          |


**Note**: We find that it is unstable to train NAS-FPN and there is a small chance that results can be 3% mAP lower.
