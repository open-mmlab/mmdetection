# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52.5 AP on the COCO detection zero-shot transfer benchmark, i.e., without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean 26.1 AP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

## COCO Results and Models

|        Model        | Backbone |   Style   |  COCO mAP  |    Pre-Train Data     |                             Config                             |                                                      Download                                                      |
| :-----------------: | :------: | :-------: | :--------: | :-------------------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |    46.7    |         O365          |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |    48.1    |      O365,GoldG       |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |    48.4    |   O365,GoldG,Cap4M    | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 48.5(+1.8) |         O365          |                          [config](<>)                          |                                                    [model](<>)                                                     |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.4(+2.3) |      O365,GoldG       |                          [config](<>)                          |                                                    [model](<>)                                                     |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.5(+2.1) |    O365,GoldG,GRIT    |                          [config](<>)                          |                                                    [model](<>)                                                     |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.4(+2.0) | O365,GoldG,GRIT,V3Det |                          [config](<>)                          |                                                    [model](<>)                                                     |

## LVIS Results

|        Model        | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |    Pre-Train Data     |    Config    |  Download   |
| :-----------------: | :---------: | :---------: | :---------: | :---------: | :--------: | :--------: | :--------: | :---------: | :-------------------: | :----------: | :---------: |
|  Grounding DINO-T   |    18.8     |    24.2     |    34.7     |    28.8     |    10.1    |    15.3    |    29.9    |    20.1     |   O365,GoldG,Cap4M    | [config](<>) | [model](<>) |
| Grounding DINO-T-V2 |    28.1     |    30.2     |    42.0     | 35.7(+6.9)  |    17.1    |    22.4    |    36.5    | 27.0(+6.9)  |      O365,GoldG       | [config](<>) | [model](<>) |
| Grounding DINO-T-V2 |    26.6     |    32.4     |    41.8     | 36.5(+7.7)  |    17.3    |    22.6    |    36.4    | 27.1(+7.0)  |    O365,GoldG,GRIT    | [config](<>) | [model](<>) |
| Grounding DINO-T-V2 |    34.2     |    37.4     |    46.2     | 41.4(+12.6) |    23.6    |    27.6    |    40.5    | 31.9(+11.8) | O365,GoldG,GRIT,V3Det | [config](<>) | [model](<>) |

## ODinW (Object Detection in the Wild) Results

Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets and tasks. However, it remains challenging to evaluate the transferablity of these models due to the lack of easy-to-use evaluation toolkits and public benchmarks. To tackle this, we build ELEVATER 1 , the first benchmark and toolkit for evaluating (pre-trained) language-augmented visual models. ELEVATER is composed of three components. (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to facilitate model evaluation on downstream tasks. (iii) Metrics. A variety of evaluation metrics are used to measure sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). ELEVATER is platform for Computer Vision in the Wild (CVinW), and is publicly released at https://computer-vision-in-the-wild.github.io/ELEVATER/

### Results and models of ODinW13

| Method                | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------- | ---------------------------------------- | ------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| AerialMaritimeDrone   | 0.173                                    | 0.133                                 | 0.155                                      | 0.151                                            |
| Aquarium              | 0.195                                    | 0.252                                 | 0.261                                      | 0.283                                            |
| CottontailRabbits     | 0.799                                    | 0.771                                 | 0.810                                      | 0.786                                            |
| EgoHands              | 0.608                                    | 0.499                                 | 0.537                                      | 0.519                                            |
| NorthAmericaMushrooms | 0.507                                    | 0.331                                 | 0.462                                      | 0.767                                            |
| Packages              | 0.687                                    | 0.707                                 | 0.687                                      | 0.706                                            |
| PascalVOC             | 0.563                                    | 0.565                                 | 0.580                                      | 0.566                                            |
| pistols               | 0.726                                    | 0.585                                 | 0.709                                      | 0.729                                            |
| pothole               | 0.215                                    | 0.136                                 | 0.285                                      | 0.243                                            |
| Raccoon               | 0.549                                    | 0.469                                 | 0.511                                      | 0.535                                            |
| ShellfishOpenImages   | 0.393                                    | 0.321                                 | 0.437                                      | 0.488                                            |
| thermalDogsAndPeople  | 0.657                                    | 0.556                                 | 0.603                                      | 0.542                                            |
| VehiclesOpenImages    | 0.613                                    | 0.566                                 | 0.603                                      | 0.615                                            |
| Average               | **0.514**                                | **0.453**                             | **0.511**                                  | **0.533**                                        |

### Results and models of ODinW35

| Method                      | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------------- | ---------------------------------------- | ------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| AerialMaritimeDrone_large   | 0.173                                    | 0.133                                 | 0.155                                      | 0.151                                            |
| AerialMaritimeDrone_tiled   | 0.206                                    | 0.170                                 | 0.225                                      | 0.206                                            |
| AmericanSignLanguageLetters | 0.002                                    | 0.016                                 | 0.020                                      | 0.007                                            |
| Aquarium                    | 0.195                                    | 0.252                                 | 0.261                                      | 0.283                                            |
| BCCD                        | 0.161                                    | 0.069                                 | 0.118                                      | 0.077                                            |
| boggleBoards                | 0.000                                    | 0.002                                 | 0.001                                      | 0.002                                            |
| brackishUnderwater          | 0.021                                    | 0.033                                 | 0.021                                      | 0.025                                            |
| ChessPieces                 | 0.000                                    | 0.000                                 | 0.000                                      | 0.000                                            |
| CottontailRabbits           | 0.806                                    | 0.771                                 | 0.810                                      | 0.786                                            |
| dice                        | 0.004                                    | 0.002                                 | 0.005                                      | 0.001                                            |
| DroneControl                | 0.042                                    | 0.047                                 | 0.097                                      | 0.074                                            |
| EgoHands_generic            | 0.608                                    | 0.527                                 | 0.537                                      | 0.519                                            |
| EgoHands_specific           | 0.002                                    | 0.001                                 | 0.005                                      | 0.003                                            |
| HardHatWorkers              | 0.046                                    | 0.048                                 | 0.070                                      | 0.108                                            |
| MaskWearing                 | 0.004                                    | 0.009                                 | 0.004                                      | 0.009                                            |
| MountainDewCommercial       | 0.430                                    | 0.453                                 | 0.465                                      | 0.430                                            |
| NorthAmericaMushrooms       | 0.471                                    | 0.331                                 | 0.462                                      | 0.767                                            |
| openPoetryVision            | 0.000                                    | 0.001                                 | 0.000                                      | 0.000                                            |
| OxfordPets_by_breed         | 0.003                                    | 0.002                                 | 0.004                                      | 0.004                                            |
| OxfordPets_by_species       | 0.011                                    | 0.019                                 | 0.016                                      | 0.015                                            |
| PKLot                       | 0.001                                    | 0.004                                 | 0.002                                      | 0.007                                            |
| Packages                    | 0.695                                    | 0.707                                 | 0.687                                      | 0.706                                            |
| PascalVOC                   | 0.563                                    | 0.565                                 | 0.580                                      | 0.566                                            |
| pistols                     | 0.726                                    | 0.585                                 | 0.709                                      | 0.729                                            |
| plantdoc                    | 0.005                                    | 0.005                                 | 0.007                                      | 0.011                                            |
| pothole                     | 0.215                                    | 0.136                                 | 0.219                                      | 0.168                                            |
| Raccoons                    | 0.549                                    | 0.469                                 | 0.511                                      | 0.535                                            |
| selfdrivingCar              | 0.089                                    | 0.091                                 | 0.076                                      | 0.083                                            |
| ShellfishOpenImages         | 0.393                                    | 0.321                                 | 0.437                                      | 0.488                                            |
| ThermalCheetah              | 0.087                                    | 0.063                                 | 0.081                                      | 0.045                                            |
| thermalDogsAndPeople        | 0.657                                    | 0.556                                 | 0.603                                      | 0.543                                            |
| UnoCards                    | 0.006                                    | 0.012                                 | 0.010                                      | 0.005                                            |
| VehiclesOpenImages          | 0.613                                    | 0.566                                 | 0.603                                      | 0.615                                            |
| WildfireSmoke               | 0.134                                    | 0.106                                 | 0.154                                      | 0.127                                            |
| websiteScreenshots          | 0.012                                    | 0.02                                  | 0.016                                      | 0.016                                            |
| Average                     | **0.227**                                | **0.202**                             | **0.228**                                  | **0.284**                                        |

## Referring Expression Comprehension Results

| Method                                  | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------------------------- | ---------------------------------------- | ------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| RefCOCO val @1,5,10                     | 50.77/89.45/94.86                        | 53.06/89.91/94.69                     | 53.4/90.3/95.5                             | 53.1/89.7/95.1                                   |
| RefCOCO testA @1,5,10                   | 57.45/91.29/95.62                        | 59.70/91.50/95.88                     | 58.8/91.70/96.2                            | 59.1/91.0/95.5                                   |
| RefCOCO testB @1,5,10                   | 44.97/86.54/92.88                        | 46.38/86.87/92.21                     | 46.8/87.7/93.3                             | 46.8/87.8/93.6                                   |
| RefCOCO+ val @1,5,10                    | 51.64/86.35/92.57                        | 53.11/87.00/92.79                     | 53.5/88.00/93.7                            | 52.7/87.7/93.5                                   |
| RefCOCO+ testA @1,5,10                  | 57.25/86.74/92.65                        | 58.94/87.34/92.91                     | 59.0/88.1/93.7                             | 58.7/87.2/93.1                                   |
| RefCOCO+ testB @1,5,10                  | 46.35/84.05/90.67                        | 47.92/84.31/91.04                     | 47.9/85.5/92.7                             | 48.4/85.8/92.1                                   |
| RefCOCOg val @1,5,10                    | 60.42/92.10/96.18                        | 61.23/92.61/96.14                     | 62.7/93.3/97.0                             | 62.9/93.3/97.2                                   |
| RefCOCOg test @1,5,10                   | 59.74/92.08/96.28                        | 61.13/93.26/96.72                     | 62.6/94.9/97.1                             | 62.9/93.9/97.43                                  |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 41.32/91.82                              | 39.76/84.65                           | 40.7/89.7                                  | 41.0/91.3                                        |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 27.23/90.24                              | 26.25/89.04                           | 26.0/91.9                                  | 26.1/93.0                                        |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 29.70/93.49                              | 31.31/84.79                           | 30.6/90.2                                  | 30.4/92.3                                        |

## Description Detection Dataset

```shell
pip install ddd-dataset
```

| Method                           | mode     | Grounding DINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT,V3Det) |
| -------------------------------- | -------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| FULL/short/middle/long/very long | concat   | 17.2/18.0/18.7/14.8/16.3                  | 15.6/17.3/16.7/14.3/13.1              | 17.0/17.7/18.0/15.7/15.7                   | 17.5/23.4/18.3/14.7/13.8                         |
| FULL/short/middle/long/very long | parallel | 22.3/28.2/24.8/19.1/13.9                  |                                       | 22.5/25.6/25.1/20.5/14.9                   | 22.9/28.1/25.4/20.4/14.4                         |
| PRES/short/middle/long/very long | concat   | 17.8/18.3/19.2/15.2/17.3                  | 16.4/18.4/17.3/14.5/14.2              | 17.9/19.0/18.3/16.5/17.5                   | 18.0/23.7/18.6/15.4/13.3                         |
| PRES/short/middle/long/very long | parallel | 21.0/27.0/22.8/17.5/12.5                  |                                       | 21.5/25.2/23.0/19.0/15.0                   | 21.9/27.4/23.2/19.1/14.2                         |
| ABS/short/middle/long/very long  | concat   | 15.4/17.1/16.4/13.6/14.9                  | 13.4/13.4/14.5/13.5/11.9              | 14.5/13.1/16.7/13.6/13.3                   | 15.9/22.2/17.1/12.5/14.4                         |
| ABS/short/middle/long/very long  | parallel | 26.0/32.0/33.0/23.6/15.5                  |                                       | 25.6/26.8/33.9/24.5/14.7                   | 26.0/30.3/34.1/23.9/14.6                         |

Note:

1. Considering that the evaluation time for Inter-scenario is very long and the performance is low, it is temporarily not supported. The mentioned metrics are for Intra-scenario.
2. `concat` is the default inference mode for Grounding DINO, where it concatenates multiple sub-sentences with "." to form a single sentence for inference. On the other hand, "parallel" performs inference on each sub-sentence in a for-loop.

## Flickr30k Results

|        Model        |    Pre-Train Data     | Val R@1 | Val R@5 | Val R@10 | Tesst R@1 | Test R@5 | Test R@10 |                          Config                           |                                                                                                                                                                                                                                         Download                                                                                                                                                                                                                                          |
| :-----------------: | :-------------------: | ------- | ------- | -------- | --------- | -------- | --------- | :-------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Grounding DINO-T   |   O365,GoldG,Cap4M    | 87.8    | 96.6    | 98.0     | 88.1      | 96.9     | 98.2      | [config](grounding_dino_swin-t_finetune_16xb2_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth)                                                                                                \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544.log.json) |
| Grounding DINO-T-V2 |      O365,GoldG       |         |         |          |           |          |           |                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Grounding DINO-T-V2 |    O365,GoldG,GRIT    |         |         |          |           |          |           |                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Grounding DINO-T-V2 | O365,GoldG,GRIT,V3Det |         |         |          |           |          |           |                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

Note:

1. `@1,5,10` refers to precision at the top 1, 5, and 10 positions in a predicted ranked list.
