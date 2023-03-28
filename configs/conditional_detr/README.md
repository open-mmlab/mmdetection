# Conditional DETR

> [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152)

<!-- [ALGORITHM] -->

## Abstract

The DETR approach applies the transformer encoder and decoder architecture to object detection and achieves promising performance. In this paper, we handle the critical issue, slow training convergence, and present a conditional cross-attention mechanism for fast DETR training. Our approach is motivated by that the cross-attention in DETR relies highly on the content embeddings and that the spatial embeddings make minor contributions, increasing the need for high-quality content embeddings and thus increasing the training difficulty.

<div align=center>
<img src="https://github.com/Atten4Vis/ConditionalDETR/blob/main/.github/attention-maps.png?raw=true"/>
</div>

Our conditional DETR learns a conditional spatial query from the decoder embedding for decoder multi-head cross-attention. The benefit is that through the conditional spatial query, each cross-attention head is able to attend to a band containing a distinct region, e.g., one object extremity or a region inside the object box (Figure 1). This narrows down the spatial range for localizing the distinct regions for object classification and box regression, thus relaxing the dependence on the content embeddings and easing the training. Empirical results show that conditional DETR converges 6.7x faster for the backbones R50 and R101 and 10x faster for stronger backbones DC5-R50 and DC5-R101.

<div align=center>
<img src="https://github.com/Atten4Vis/ConditionalDETR/raw/main/.github/conditional-detr.png" width="48%"/>
<img src="https://github.com/Atten4Vis/ConditionalDETR/raw/main/.github/convergence-curve.png" width="48%"/>
</div>

## Results and Models

We provide the config files and models for Conditional DETR: [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152).

| Backbone |      Model       | Lr schd | Mem (GB) | Inf time (fps) | box AP |                      Config                       |                                                                                                                                                                        Download                                                                                                                                                                        |
| :------: | :--------------: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | Conditional DETR |   50e   |          |                |  41.1  | [config](./conditional-detr_r50_8xb2-50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/conditional_detr/conditional-detr_r50_8xb2-50e_coco/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/conditional_detr/conditional-detr_r50_8xb2-50e_coco/conditional-detr_r50_8xb2-50e_coco_20221121_180202.log.json) |

## Citation

```latex
@inproceedings{meng2021-CondDETR,
  title       = {Conditional DETR for Fast Training Convergence},
  author      = {Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year        = {2021}
}
```
