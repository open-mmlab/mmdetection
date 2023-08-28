# DDQ

> [Dense Distinct Query for End-to-End Object Detection](https://arxiv.org/abs/2303.12776)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

One-to-one label assignment in object detection has successfully obviated the need for non-maximum suppression (NMS) as postprocessing and makes the pipeline end-to-end. However, it triggers a new dilemma as the widely used sparse queries cannot guarantee a high recall, while dense queries inevitably bring more similar queries and encounter optimization difficulties. As both sparse and dense queries are problematic, then what are the expected queries in end-to-end object detection? This paper shows that the solution should be Dense Distinct Queries (DDQ). Concretely, we first lay dense queries like traditional detectors and then select distinct ones for one-to-one assignments. DDQ blends the advantages of traditional and recent end-to-end detectors and significantly improves the performance of various detectors including FCN, R-CNN, and DETRs. Most impressively, DDQ-DETR achieves 52.1 AP on MS-COCO dataset within 12 epochs using a ResNet-50 backbone, outperforming all existing detectors in the same setting. DDQ also shares the benefit of end-to-end detectors in crowded scenes and achieves 93.8 AP on CrowdHuman. We hope DDQ can inspire researchers to consider the complementarity between traditional methods and end-to-end detectors.

![ddq_arch](https://github.com/open-mmlab/mmdetection/assets/33146359/5ca9f11b-b6f3-454f-a2d1-3009ee337bbc)

## Results and Models

|       Model       | Backbone | Lr schd | Augmentation | box AP(val) |                       Config                       |                                                                                                                                                         Download                                                                                                                                                         |
| :---------------: | :------: | :-----: | :----------: | :---------: | :------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  DDQ DETR-4scale  |   R-50   |   12e   |     DETR     |    51.4     |  [config](./ddq-detr-4scale_r50_8xb2-12e_coco.py)  | [model](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711.log.json) |
| DDQ DETR-5scale\* |   R-50   |   12e   |     DETR     |    52.1     |  [config](./ddq-detr-5scale_r50_8xb2-12e_coco.py)  |                                                            [model](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_5scale_coco_1x.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_5scale_coco_1x_20230319_103307.log)                                                            |
| DDQ DETR-4scale\* |  Swin-L  |   30e   |     DETR     |    58.7     | [config](./ddq-detr-4scale_swinl_8xb2-30e_coco.py) |                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e_20230316_221721_20230318_143554.log)                                                         |

**Note**

- Models labeled * are not trained by us, but from [DDQ official website](https://github.com/jshilong/DDQ).
- We find that the performance is unstable and may fluctuate by about 0.2 mAP.

## Citation

```latex
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Shilong and Wang, Xinjiang and Wang, Jiaqi and Pang, Jiangmiao and Lyu, Chengqi and Zhang, Wenwei and Luo, Ping and Chen, Kai},
    title     = {Dense Distinct Query for End-to-End Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7329-7338}
}
```
