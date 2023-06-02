# CrowdDet

> [Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://arxiv.org/abs/2003.09163)

<!-- [ALGORITHM] -->

## Abstract

We propose a simple yet effective proposal-based object detector, aiming at detecting highly-overlapped instances in crowded scenes. The key of our approach is to let each proposal predict a set of correlated instances rather than a single one in previous proposal-based frameworks. Equipped with new techniques such as EMD Loss and Set NMS, our detector can effectively handle the difficulty of detecting highly overlapped objects. On a FPN-Res50 baseline, our detector can obtain 4.9% AP gains on challenging CrowdHuman dataset and 1.0% MR^−2 improvements on CityPersons dataset, without bells and whistles. Moreover, on less crowed datasets like COCO, our approach can still achieve moderate improvement, suggesting the proposed method is robust to crowdedness. Code and pre-trained models will be released at https://github.com/megvii-model/CrowdDetection.

<div align=center>
<img src="https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg"/>
</div>

## Results and Models

| Backbone |  RM   |  Style  | Mem (GB) | Inf time (fps) | box AP |                             Config                              |                                                                                                                                                                                            Download                                                                                                                                                                                            |
| :------: | :---: | :-----: | :------: | :------------: | :----: | :-------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN | False | pytorch |   4.4    |       -        |  90.0  |    [config](./crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py)     |               [model](https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman_20221023_174954-dc319c2d.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman_20221023_174954.log.json)               |
| R-50-FPN | True  | pytorch |   4.8    |       -        | 90.32  | [config](./crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman_20221024_215917-45602806.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman_20221024_215917.log.json) |

Note:

- RM indicates whether to use the refine module.
- The dataset for training and testing this model is `CrowdHuman`, and the metric of `box AP` is calculated by `mmdet/evaluation/metrics/crowdhuman_metric.py`.

## Citation

```latex
@inproceedings{Chu_2020_CVPR,
  title={Detection in Crowded Scenes: One Proposal, Multiple Predictions},
  author={Chu, Xuangeng and Zheng, Anlin and Zhang, Xiangyu and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
