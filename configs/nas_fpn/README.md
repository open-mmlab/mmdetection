# NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection

## Introduction

[ALGORITHM]

```latex
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

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50-FPN    | 50e     | 12.9     | 22.9           | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco_20200529_095329.log.json) |
| R-50-NASFPN | 50e     | 13.2     | 23.0           | 40.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco_20200528_230008.log.json) |

**Note**: We find that it is unstable to train NAS-FPN and there is a small chance that results can be 3% mAP lower.
