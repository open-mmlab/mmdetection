# BoxInst

> [BoxInst: High-Performance Instance Segmentation with Box Annotations](https://arxiv.org/pdf/2012.02310.pdf)

<!-- [ALGORITHM] -->

## Abstract

We present a high-performance method that can achieve mask-level instance segmentation with only bounding-box annotations for training. While this setting has been studied in the literature, here we show significantly stronger performance with a simple design (e.g., dramatically improving previous best reported mask AP of 21.1% to 31.6% on the COCO dataset). Our core idea is to redesign the loss
of learning masks in instance segmentation, with no modification to the segmentation network itself. The new loss functions can supervise the mask training without relying on mask annotations. This is made possible with two loss terms, namely, 1) a surrogate term that minimizes the discrepancy between the projections of the ground-truth box and the predicted mask; 2) a pairwise loss that can exploit the prior that proximal pixels with similar colors are very likely to have the same category label. Experiments demonstrate that the redesigned mask loss can yield surprisingly high-quality instance masks with only box annotations. For example, without using any mask annotations, with a ResNet-101 backbone and 3× training schedule, we achieve 33.2% mask AP on COCO test-dev split (vs. 39.1% of the fully supervised counterpart). Our excellent experiment results on COCO and Pascal VOC indicate that our method dramatically narrows the performance gap between weakly and fully supervised instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/57584090/209087723-756b76d7-5061-4000-a93c-df1194a439a0.png"/>
</div>

## Results and Models

| Backbone |  Style  | MS train | Lr schd | bbox AP | mask AP |                   Config                    |                                                                                                                                                  Download                                                                                                                                                   |
| :------: | :-----: | :------: | :-----: | :-----: | :-----: | :-----------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |    Y     |   1x    |  39.6   |  31.1   | [config](./boxinst_r50_fpn_ms-90k_coco.py)  |  [model](https://download.openmmlab.com/mmdetection/v3.0/boxinst/boxinst_r50_fpn_ms-90k_coco/boxinst_r50_fpn_ms-90k_coco_20221228_163052-6add751a.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/boxinst/boxinst_r50_fpn_ms-90k_coco/boxinst_r50_fpn_ms-90k_coco_20221228_163052.log.json)   |
|  R-101   | pytorch |    Y     |   1x    |  41.8   |  32.7   | [config](./boxinst_r101_fpn_ms-90k_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/boxinst/boxinst_r101_fpn_ms-90k_coco/boxinst_r101_fpn_ms-90k_coco_20221229_145106-facf375b.pth) \|[log](https://download.openmmlab.com/mmdetection/v3.0/boxinst/boxinst_r101_fpn_ms-90k_coco/boxinst_r101_fpn_ms-90k_coco_20221229_145106.log.json) |

## Citation

```latex
@inproceedings{tian2020boxinst,
  title     =  {{BoxInst}: High-Performance Instance Segmentation with Box Annotations},
  author    =  {Tian, Zhi and Shen, Chunhua and Wang, Xinlong and Chen, Hao},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2021}
}
```
