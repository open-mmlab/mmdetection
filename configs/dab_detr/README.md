# DAB-DETR

> [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329)

<!-- [ALGORITHM] -->

## Abstract

We present in this paper a novel query formulation using dynamic anchor boxes for DETR (DEtection TRansformer) and offer a deeper understanding of the role of queries in DETR. This new formulation directly uses box coordinates as queries in Transformer decoders and dynamically updates them layer-by-layer. Using box coordinates not only helps using explicit positional priors to improve the query-to-feature similarity and eliminate the slow training convergence issue in DETR, but also allows us to modulate the positional attention map using the box width and height information. Such a design makes it clear that queries in DETR can be implemented as performing soft ROI pooling layer-by-layer in a cascade manner. As a result, it leads to the best performance on MS-COCO benchmark among the DETR-like detection models under the same setting, e.g., AP 45.7% using ResNet50-DC5 as backbone trained in 50 epochs. We also conducted extensive experiments to confirm our analysis and verify the effectiveness of our methods.

<div align=center>
<img src="https://github.com/IDEA-Research/DAB-DETR/blob/main/figure/arch.png?raw=true"/>
</div>
<div align=center>
<img src="https://github.com/IDEA-Research/DAB-DETR/blob/main/figure/model.png?raw=true"/>
</div>
<div align=center>
<img src="https://github.com/IDEA-Research/DAB-DETR/blob/main/figure/results.png?raw=true"/>
</div>

## Results and Models

We provide the config files and models for DAB-DETR: [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329).

| Backbone |  Model   | Lr schd | Mem (GB) | Inf time (fps) | box AP |                  Config                   |                                                                                                                                                Download                                                                                                                                                |
| :------: | :------: | :-----: | :------: | :------------: | :----: | :---------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | DAB-DETR |   50e   |          |                |  42.3  | [config](./dab-detr_r50_8xb2-50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/dab_detr/dab-detr_r50_8xb2-50e_coco/dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dab_detr/dab-detr_r50_8xb2-50e_coco/dab-detr_r50_8xb2-50e_coco_20221122_120837.log.json) |

## Citation

```latex
@inproceedings{
  liu2022dabdetr,
  title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
  author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```
