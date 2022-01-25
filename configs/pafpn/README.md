# PAFPN

> [Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)

<!-- [ALGORITHM] -->

## Abstract

The way that information propagates in neural networks is of great importance. In this paper, we propose Path Aggregation Network (PANet) aiming at boosting information flow in proposal-based instance segmentation framework. Specifically, we enhance the entire feature hierarchy with accurate localization signals in lower layers by bottom-up path augmentation, which shortens the information path between lower layers and topmost feature. We present adaptive feature pooling, which links feature grid and all feature levels to make useful information in each feature level propagate directly to following proposal subnetworks. A complementary branch capturing different views for each proposal is created to further improve mask prediction. These improvements are simple to implement, with subtle extra computational overhead. Our PANet reaches the 1st place in the COCO 2017 Challenge Instance Segmentation task and the 2nd place in Object Detection task without large-batch training. It is also state-of-the-art on MVD and Cityscapes.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143968947-5f2d7e8a-a236-4d59-8f2d-7fbb12764845.png"/>
</div>

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN      | pytorch    | 1x      | 4.0      | 17.2           | 37.5   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_20200503_105836.log.json) |

## Citation

```latex
@inproceedings{liu2018path,
  author = {Shu Liu and
            Lu Qi and
            Haifang Qin and
            Jianping Shi and
            Jiaya Jia},
  title = {Path Aggregation Network for Instance Segmentation},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
