# SABL

> [Side-Aware Boundary Localization for More Precise Object Detection](https://arxiv.org/abs/1912.04260)

<!-- [ALGORITHM] -->

## Abstract

Current object detection frameworks mainly rely on bounding box regression to localize objects. Despite the remarkable progress in recent years, the precision of bounding box regression remains unsatisfactory, hence limiting performance in object detection. We observe that precise localization requires careful placement of each side of the bounding box. However, the mainstream approach, which focuses on predicting centers and sizes, is not the most effective way to accomplish this task, especially when there exists displacements with large variance between the anchors and the targets. In this paper, we propose an alternative approach, named as Side-Aware Boundary Localization (SABL), where each side of the bounding box is respectively localized with a dedicated network branch. To tackle the difficulty of precise localization in the presence of displacements with large variance, we further propose a two-step localization scheme, which first predicts a range of movement through bucket prediction and then pinpoints the precise position within the predicted bucket. We test the proposed method on both two-stage and single-stage detection frameworks. Replacing the standard bounding box regression branch with the proposed design leads to significant improvements on Faster R-CNN, RetinaNet, and Cascade R-CNN, by 3.0%, 1.7%, and 0.9%, respectively.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143973698-3dfaea91-4415-4818-9781-5017183e7489.png"/>
</div>

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).
Single-scale testing (1333x800) is adopted in all results.

|       Method       | Backbone  | Lr schd | ms-train | box AP |                                                       Config                                                       |                                                                                                                                  Download                                                                                                                                   |
| :----------------: | :-------: | :-----: | :------: | :----: | :----------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL Faster R-CNN  | R-50-FPN  |   1x    |    N     |  39.9  |  [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py)  |    [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/20200830_130324.log.json)    |
| SABL Faster R-CNN  | R-101-FPN |   1x    |    N     |  41.7  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_faster_rcnn_r101_fpn_1x_coco.py)  |  [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/sabl_faster_rcnn_r101_fpn_1x_coco-f804c6c1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/20200830_183949.log.json)   |
| SABL Cascade R-CNN | R-50-FPN  |   1x    |    N     |  41.6  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco.py)  |  [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/sabl_cascade_rcnn_r50_fpn_1x_coco-e1748e5e.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/20200831_033726.log.json)   |
| SABL Cascade R-CNN | R-101-FPN |   1x    |    N     |  43.0  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/sabl_cascade_rcnn_r101_fpn_1x_coco-2b83e87c.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/20200831_141745.log.json) |

|     Method     | Backbone  | GN  | Lr schd |  ms-train   | box AP |                                                            Config                                                             |                                                                                                                                                   Download                                                                                                                                                   |
| :------------: | :-------: | :-: | :-----: | :---------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL RetinaNet | R-50-FPN  |  N  |   1x    |      N      |  37.7  |        [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py)         |                       [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/20200830_053451.log.json)                        |
| SABL RetinaNet | R-50-FPN  |  Y  |   1x    |      N      |  38.8  |       [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r50_fpn_gn_1x_coco.py)       |                   [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/20200831_141955.log.json)                   |
| SABL RetinaNet | R-101-FPN |  N  |   1x    |      N      |  39.7  |        [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_1x_coco.py)        |                      [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/sabl_retinanet_r101_fpn_1x_coco-42026904.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/20200831_034256.log.json)                      |
| SABL RetinaNet | R-101-FPN |  Y  |   1x    |      N      |  40.5  |      [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_1x_coco.py)       |                 [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/20200830_201422.log.json)                  |
| SABL RetinaNet | R-101-FPN |  Y  |   2x    | Y (640~800) |  42.9  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/20200830_144807.log.json) |
| SABL RetinaNet | R-101-FPN |  Y  |   2x    | Y (480~960) |  43.6  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/20200830_164537.log.json) |

## Citation

We provide config files to reproduce the object detection results in the ECCV 2020 Spotlight paper for [Side-Aware Boundary Localization for More Precise Object Detection](https://arxiv.org/abs/1912.04260).

```latex
@inproceedings{Wang_2020_ECCV,
    title = {Side-Aware Boundary Localization for More Precise Object Detection},
    author = {Jiaqi Wang and Wenwei Zhang and Yuhang Cao and Kai Chen and Jiangmiao Pang and Tao Gong and Jianping Shi and Chen Change Loy and Dahua Lin},
    booktitle = {ECCV},
    year = {2020}
}
```
