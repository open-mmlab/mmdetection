# Group DETR

> [FAST DETR TRAINING WITH GROUPWISE ONE-TO-MANY ASSIGNMENT](https://arxiv.org/abs/2207.13085)

<!-- [ALGORITHM] -->

## Abstract

Detection Transformer (DETR) relies on One-to-One assignment, i.e., assigning one ground-truth object to only one positive object query, for end-to-end object detection and lacks the capability of exploiting multiple positive object queries. Group DETR is proposed to support Group-wise One-to-Many assignment. It makes simple modifications during training: (i) adopt K groups of object queries; (ii) conduct decoder self-attention on each group of object queries with the same parameters; (iii) perform One-to-One label assignment for each group, leading to K positive object queries for each ground-truth object. In inference, It only uses one group of object queries, making no modifications to DETR architecture and processes. It is effective on DETR variants, including Conditional DETR, DAB-DETR, DN-DETR, and DINO.

<div align=center>
<img src="https://user-images.githubusercontent.com/70597027/220554743-bc1b8784-d01a-444f-ab06-5501c4a89c5d.png" width="95%"/>
<img src="https://user-images.githubusercontent.com/70597027/220554770-c13a5f8d-621d-4197-9ac9-1e8d62b423ff.png" width="95%"/>
</div>

## Results and Models

We provide the config files and models for Group DETR: [FAST DETR TRAINING WITH GROUPWISE ONE-TO-MANY ASSIGNMENT](https://arxiv.org/abs/2207.13085).

| Backbone |   Model    | Lr schd | Mem (GB) | Inf time (fps) | box AP |                       Config                        |         Download         |
| :------: | :--------: | :-----: | :------: | :------------: | :----: | :-------------------------------------------------: | :----------------------: |
|   R-50   | Group DETR |   50e   |          |                |        | [config](./configs/group_detr_r50_8xb2-50e_coco.py) | [model](<>) \| [log](<>) |

Please note that there is no extra computational cost during inference when compared to original Conditional DETR.

## Citation

```latex
@article{chen2022group,
  title={Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment},
  author={Chen, Qiang and Chen, Xiaokang and Wang, Jian and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2207.13085},
  year={2022}
}
```
