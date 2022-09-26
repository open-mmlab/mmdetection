# MS R-CNN

> [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241)

<!-- [ALGORITHM] -->

## Abstract

Letting a deep network be aware of the quality of its own predictions is an interesting yet important problem. In the task of instance segmentation, the confidence of instance classification is used as mask quality score in most instance segmentation frameworks. However, the mask quality, quantified as the IoU between the instance mask and its ground truth, is usually not well correlated with classification score. In this paper, we study this problem and propose Mask Scoring R-CNN which contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models, and outperforms the state-of-the-art Mask R-CNN. We hope our simple and effective approach will provide a new direction for improving instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967239-3a95ae92-6443-4181-9cbc-dfe16e81b969.png"/>
</div>

## Results and Models

|   Backbone   |  style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                    Config                     |                                                                                                                                                                      Download                                                                                                                                                                       |
| :----------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :-------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50-FPN   |  caffe  |   1x    |   4.5    |                |  38.2  |  36.0   | [config](./ms-rcnn_r50-caffe_fpn_1x_coco.py)  |                  [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848.log.json)                   |
|   R-50-FPN   |  caffe  |   2x    |    -     |       -        |  38.8  |  36.3   | [config](./ms-rcnn_r50-caffe_fpn_2x_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_20200506_004738.log.json)   |
|  R-101-FPN   |  caffe  |   1x    |   6.5    |                |  40.4  |  37.6   | [config](./ms-rcnn_r101-caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_20200506_004755.log.json) |
|  R-101-FPN   |  caffe  |   2x    |    -     |       -        |  41.1  |  38.1   | [config](./ms-rcnn_r101-caffe_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_20200506_011134.log.json) |
| R-X101-32x4d | pytorch |   2x    |   7.9    |      11.0      |  41.8  |  38.7   | [config](./ms-rcnn_x101-32x4d_fpn_1x_coco.py) |                    [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206_100113.log.json)                    |
| R-X101-64x4d | pytorch |   1x    |   11.0   |      8.0       |  43.0  |  39.5   | [config](./ms-rcnn_x101-64x4d_fpn_1x_coco.py) |                    [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206_091744.log.json)                    |
| R-X101-64x4d | pytorch |   2x    |   11.0   |      8.0       |  42.6  |  39.5   | [config](./ms-rcnn_x101-64x4d_fpn_2x_coco.py) |                    [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308_012247.log.json)                    |

## Citation

```latex
@inproceedings{huang2019msrcnn,
    title={Mask Scoring R-CNN},
    author={Zhaojin Huang and Lichao Huang and Yongchao Gong and Chang Huang and Xinggang Wang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019},
}
```
