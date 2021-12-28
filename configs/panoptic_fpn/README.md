# Panoptic feature pyramid networks
## Abstract

<!-- [ABSTRACT] -->

The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes). However, current state-of-the-art methods for this joint task use separate and dissimilar networks for instance and semantic segmentation, without performing any shared computation. In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks. Our approach is to endow Mask R-CNN, a popular instance segmentation method, with a semantic segmentation branch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly, this simple baseline not only remains effective for instance segmentation, but also yields a lightweight, top-performing method for semantic segmentation. In this work, we perform a detailed study of this minimally extended version of Mask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robust and accurate baseline for both tasks. Given its effectiveness and conceptual simplicity, we hope our method can serve as a strong baseline and aid future research in panoptic segmentation.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143968979-a1593758-c9d7-44a6-a3b8-d9686ef19ce8.png" height="300"/>
</div>

<!-- [PAPER_TITLE: Panoptic Feature Pyramid Networks] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1901.02446] -->

## Citation

<!-- [ALGORITHM] -->
The base method for panoptic segmentation task.

```
@inproceedings{kirillov2018panopticfpn,
  author = {
    Alexander Kirillov,
    Ross Girshick,
    Kaiming He,
    Piotr Dollar,
  },
  title = {Panoptic Feature Pyramid Networks},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

## Dataset

PanopticFPN requires COCO and [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) dataset for training and evaluation. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | PQ   |  SQ  |  RQ  | PQ_th | SQ_th | RQ_th | PQ_st | SQ_st | RQ_st | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:----:|:----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
| R-50-FPN      | pytorch    | 1x      |   4.7    |                | 40.2 | 77.8 | 49.3 | 47.8  | 80.9  | 57.5  | 28.9  | 73.1  | 37.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153.log.json) |
| R-50-FPN      | pytorch    | 3x      |   -      |     -          | 42.5 | 78.1 | 51.7 | 50.3  | 81.5  | 60.3  | 30.7  | 73.0  | 38.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155.log.json) |
| R-101-FPN     | pytorch    | 1x      |   6.7    |                | 42.2 | 78.3 | 51.4 | 50.1  | 81.4  | 59.9  | 30.3  | 73.6  | 38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/panoptic_fpn/panoptic_fpn_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r101_fpn_1x_coco/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r101_fpn_1x_coco/panoptic_fpn_r101_fpn_1x_coco_20210820_193950.log.json) |
| R-101-FPN     | pytorch    | 3x      |   -      |     -          | 44.1 | 78.9 | 53.6 | 52.1  | 81.7  | 62.3  | 32.0  | 74.6  | 40.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/panoptic_fpn/panoptic_fpn_r101_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r101_fpn_mstrain_3x_coco/panoptic_fpn_r101_fpn_mstrain_3x_coco_20210823_114712-9c99acc4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r101_fpn_mstrain_3x_coco/panoptic_fpn_r101_fpn_mstrain_3x_coco_20210823_114712.log.json) |
