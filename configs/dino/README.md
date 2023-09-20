# DINO

> [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

<!-- [ALGORITHM] -->

## Abstract

We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/207820666-099e6a85-59c4-45d6-a687-91b5781d11cd.png"/>
</div>

## Results and Models

| Backbone |    Model    | Lr schd | Better-Hyper | box AP |                        Config                         |                                                                                                                                                                    Download                                                                                                                                                                    |
| :------: | :---------: | :-----: | :----------: | :----: | :---------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | DINO-4scale |   12e   |    False     |  49.0  |     [config](./dino-4scale_r50_8xb2-12e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705.log.json)                   |
|   R-50   | DINO-4scale |   12e   |     True     |  50.1  | [config](./dino-4scale_r50_improved_8xb2-12e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607.log.json) |
|  Swin-L  | DINO-5scale |   12e   |    False     |  57.2  |    [config](./dino-5scale_swin-l_8xb2-12e_coco.py)    |               [model](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924.log)                |
|  Swin-L  | DINO-5scale |   36e   |    False     |  58.4  |    [config](./dino-5scale_swin-l_8xb2-36e_coco.py)    |                                                 [model](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth) \| [log](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/20230307_032359.log)                                                 |

### NOTE

The performance is unstable. `DINO-4scale` with `R-50` may fluctuate about 0.4 mAP.

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
