# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52.5 AP on the COCO detection zero-shot transfer benchmark, i.e., without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean 26.1 AP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

```
cd $MMDETROOT

wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth

python demo/image_demo.py \
	demo/demo.jpg \
	configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py \
	--weights groundingdino_swint_ogc_mmdet-822d7e9d.pth \
	--texts 'bench . car .'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/3a3bd6f1-e2ed-43d4-aa22-0bb07ee6f20b"/>
</div>

## Results and Models

|      Model       | backbone | COCO mAP |                  Pre-Train Data                  |                             Config                             |                                                        Download                                                        |
| :--------------: | :------: | :------: | :----------------------------------------------: | :------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| Grounding DINO-T |  Swin-T  |   48.5   |                 O365,GoldG,Cap4M                 | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) |   [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)   |
| Grounding DINO-B |  Swin-B  |   56.9   | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO |     [config](grounding_dino_swin-b_pretrain_mixeddata.py)      | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth) |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/groundingdino_to_mmdet.py). We have not retrained the model for the time being.
