# DETRs with Hybrid Matching

> [DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)

<!-- [ALGORITHM] -->

## Abstract

One-to-one set matching is a key design for DETR to establish its end-to-end capability, so that object detection does not require a hand-crafted NMS (non-maximum suppression) method to remove duplicate detections. This end-to-end signature is important for the versatility of DETR, and it has been generalized to a wide range of visual problems, including instance/semantic segmentation, human pose estimation, and point cloud/multi-view-images based detection, etc. However, we note that because there are too few queries assigned as positive samples, the one-to-one set matching significantly reduces the training efficiency of positive samples. This paper proposes a simple yet effective method based on a hybrid matching scheme that combines the original one-to-one matching branch with auxiliary queries that use one-to-many matching loss during training. This hybrid strategy has been shown to significantly improve training efficiency and improve accuracy. In inference, only the original one-to-one match branch is used, thus maintaining the end-to-end merit and the same inference efficiency of DETR. The method is named -DETR, and it shows that a wide range of representative DETR methods can be consistently improved across a wide range of visual tasks, including Deformable-DETR, 3DETR/PETRv2, PETR, and TransTrack, among others. Code will be available at: this https URL

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143877617-ad9b24fd-77ce-46aa-9689-1a44b5594132.png"/>
</div>

## Results and Models

| Backbone |           Model            | Lr schd | box AP |                                                               Config                                                                |                                                                                                                                                                                       Download                                                                                                                                                                                        |
| :------: | :------------------------: | :-----: | :----: | :---------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |  Deformable DETR + trick   |   1x    |  47.1  |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/h_deformable_detr/deformable_detr_r50_trick_1x_coco.py)    |               [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.log.json)               |
|   R-50   | DETRs with Hybrid Matching |   1x    |  48.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.log.json) |

## Citation

We provide the config files for DETRs with Hybrid Matching: [DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080).

```latex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}
```
