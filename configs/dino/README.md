# DINO

> [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

<!-- [ALGORITHM] -->

## Abstract

We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/207820666-099e6a85-59c4-45d6-a687-91b5781d11cd.png"/>
</div>

## Results and Models

| Backbone | Model | Lr schd | box AP |                    Config                    | Download |
| :------: | :---: | :-----: | :----: | :------------------------------------------: | :------: |
|   R-50   | DINO  |   12e   |  49.0  | [config](./dino_4scale_r50_8xb2-12e_coco.py) |          |

## Citation

We provide the config files for DINO: [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605).

```latex
@misc{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
  year={2022},
  eprint={2203.03605},
  archivePrefix={arXiv},
  primaryClass={cs.CV}}
```
