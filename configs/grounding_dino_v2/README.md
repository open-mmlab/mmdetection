# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52.5 AP on the COCO detection zero-shot transfer benchmark, i.e., without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean 26.1 AP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

|        Model        | Backbone |   Style   |  COCO mAP  | Official COCO mAP |    Pre-Train Data     |                             Config                             |                                                      Download                                                      |
|:-------------------:|:--------:|:---------:|:----------:|:-----------------:|:---------------------:|:--------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
|  Grounding DINO-T   |  Swin-T  | Zero-shot |            |       46.7        |         O365          |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |            |       48.1        |      O365,GoldG       |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |    48.5    |       48.4        |   O365,GoldG,Cap4M    | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 48.5(+1.8) |                   |         O365          |                           [config]()                           |                                                     [model]()                                                      |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.4(+2.3) |                   |      O365,GoldG       |                           [config]()                           |                                                     [model]()                                                      |


