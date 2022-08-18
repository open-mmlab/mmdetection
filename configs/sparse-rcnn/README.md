# Sparse R-CNN

> [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450)

<!-- [ALGORITHM] -->

## Abstract

We present Sparse R-CNN, a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as k anchor boxes pre-defined on all grids of image feature map of size H×W. In our method, however, a fixed sparse set of learned object proposals, total length of N, are provided to object recognition head to perform classification and location. By eliminating HWk (up to hundreds of thousands) hand-designed object candidates to N (e.g. 100) learnable proposals, Sparse R-CNN completely avoids all efforts related to object candidates design and many-to-one label assignment. More importantly, final predictions are directly output without non-maximum suppression post-procedure. Sparse R-CNN demonstrates accuracy, run-time and training convergence performance on par with the well-established detector baselines on the challenging COCO dataset, e.g., achieving 45.0 AP in standard 3× training schedule and running at 22 fps using ResNet-50 FPN model. We hope our work could inspire re-thinking the convention of dense prior in object detectors.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998489-8a5a687d-ceec-4590-8347-708e427e7dfe.png" height="300"/>
</div>

## Results and Models

|    Model     | Backbone  |  Style  | Lr schd | Number of Proposals | Multi-Scale | RandomCrop | box AP |                                                                         Config                                                                         |                                                                                                                                                                                                                                 Download                                                                                                                                                                                                                                  |
| :----------: | :-------: | :-----: | :-----: | :-----------------: | :---------: | :--------: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Sparse R-CNN | R-50-FPN  | pytorch |   1x    |         100         |    False    |   False    |  37.9  |                   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py)                   |                                                                         [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.log.json)                                                                         |
| Sparse R-CNN | R-50-FPN  | pytorch |   3x    |         100         |    True     |   False    |  42.8  |           [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py)           |                                         [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.log.json)                                         |
| Sparse R-CNN | R-50-FPN  | pytorch |   3x    |         300         |    True     |    True    |  45.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.log.json)   |
| Sparse R-CNN | R-101-FPN | pytorch |   3x    |         100         |    True     |   False    |  44.2  |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py)           |                                       [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.log.json)                                       |
| Sparse R-CNN | R-101-FPN | pytorch |   3x    |         300         |    True     |    True    |  46.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.log.json) |

### Notes

We observe about 0.3 AP noise especially when using ResNet-101 as the backbone.

## Citation

```latex
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```
