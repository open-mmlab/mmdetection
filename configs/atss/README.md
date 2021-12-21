# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

## Abstract

<!-- [ABSTRACT] -->

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, we first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a point. This shows that how to select positive and negative training samples is important for current object detectors. Then, we propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object. It significantly improves the performance of anchor-based and anchor-free detectors and bridges the gap between them. Finally, we discuss the necessity of tiling multiple anchors per location on the image to detect objects. Extensive experiments conducted on MS COCO support our aforementioned analysis and conclusions. With the newly introduced ATSS, we improve state-of-the-art detectors by a large margin to 50.7% AP without introducing any overhead.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143870776-c81168f5-e8b2-44ee-978b-509e4372c5c9.png"/>
</div>

<!-- [PAPER_TITLE: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1912.02424] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@article{zhang2019bridging,
  title   =  {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author  =  {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  journal =  {arXiv preprint arXiv:1912.02424},
  year    =  {2019}
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 3.7      | 19.7           |  39.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss/atss_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209_102539.log.json) |
| R-101     | pytorch | 1x      | 5.6      | 12.3           |  41.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss/atss_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.log.json) |
