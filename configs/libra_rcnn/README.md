# Libra R-CNN

> [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/abs/1904.02701)

<!-- [ALGORITHM] -->

## Abstract

Compared with model architectures, the training process, which is also crucial to the success of detectors, has received relatively less attention in object detection. In this work, we carefully revisit the standard training practice of detectors, and find that the detection performance is often limited by the imbalance during the training process, which generally consists in three levels - sample level, feature level, and objective level. To mitigate the adverse effects caused thereby, we propose Libra R-CNN, a simple but effective framework towards balanced learning for object detection. It integrates three novel components: IoU-balanced sampling, balanced feature pyramid, and balanced L1 loss, respectively for reducing the imbalance at sample, feature, and objective level. Benefitted from the overall balanced design, Libra R-CNN significantly improves the detection performance. Without bells and whistles, it achieves 2.5 points and 2.0 points higher Average Precision (AP) than FPN Faster R-CNN and RetinaNet respectively on MSCOCO.

Instance recognition is rapidly advanced along with the developments of various deep convolutional neural networks. Compared to the architectures of networks, the training process, which is also crucial to the success of detectors, has received relatively less attention. In this work, we carefully revisit the standard training practice of detectors, and find that the detection performance is often limited by the imbalance during the training process, which generally consists in three levels - sample level, feature level, and objective level. To mitigate the adverse effects caused thereby, we propose Libra R-CNN, a simple yet effective framework towards balanced learning for instance recognition. It integrates IoU-balanced sampling, balanced feature pyramid, and objective re-weighting, respectively for reducing the imbalance at sample, feature, and objective level. Extensive experiments conducted on MS COCO, LVIS and Pascal VOC datasets prove the effectiveness of the overall balanced design.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143966392-2633684c-a67a-4269-b71b-afe945c67bcd.png"/>
</div>

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Architecture |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                         Config                          |                                                                                                                                                                          Download                                                                                                                                                                           |
| :----------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN |    R-50-FPN     | pytorch |   1x    |   4.6    |      19.0      |  38.3  |    [config](./libra-faster-rcnn_r50_fpn_1x_coco.py)     |               [model](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json)               |
|  Fast R-CNN  |    R-50-FPN     | pytorch |   1x    |          |                |        |                                                         |                                                                                                                                                                                                                                                                                                                                                             |
| Faster R-CNN |    R-101-FPN    | pytorch |   1x    |   6.5    |      14.4      |  40.1  |    [config](./libra-faster-rcnn_r101_fpn_1x_coco.py)    |             [model](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203_001405.log.json)             |
| Faster R-CNN | X-101-64x4d-FPN | pytorch |   1x    |   10.8   |      8.5       |  42.7  | [config](./libra-faster-rcnn_x101-64x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315_231625.log.json) |
|  RetinaNet   |    R-50-FPN     | pytorch |   1x    |   4.2    |      17.7      |  37.6  |     [config](./libra-retinanet_r50_fpn_1x_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205_112757.log.json)                   |

## Citation

We provide config files to reproduce the results in the CVPR 2019 paper [Libra R-CNN](https://arxiv.org/pdf/1904.02701.pdf).

The extended version of [Libra R-CNN](https://arxiv.org/pdf/2108.10175.pdf) is accpeted by IJCV.

```latex
@inproceedings{pang2019libra,
  title={Libra R-CNN: Towards Balanced Learning for Object Detection},
  author={Pang, Jiangmiao and Chen, Kai and Shi, Jianping and Feng, Huajun and Ouyang, Wanli and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{pang2021towards,
  title={Towards Balanced Learning for Instance Recognition},
  author={Pang, Jiangmiao and Chen, Kai and Li, Qi and Xu, Zhihai and Feng, Huajun and Shi, Jianping and Ouyang, Wanli and Lin, Dahua},
  journal={International Journal of Computer Vision},
  volume={129},
  number={5},
  pages={1376--1393},
  year={2021},
  publisher={Springer}
}
```
