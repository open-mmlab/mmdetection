# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52.5 AP on the COCO detection zero-shot transfer benchmark, i.e., without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean 26.1 AP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

## COCO Results and Models

|        Model        | Backbone |   Style   |  COCO mAP  | Official COCO mAP |  Pre-Train Data  |                             Config                             |                                                      Download                                                      |
|:-------------------:|:--------:|:---------:|:----------:|:-----------------:|:----------------:|:--------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
|  Grounding DINO-T   |  Swin-T  | Zero-shot |            |       46.7        |       O365       |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |            |       48.1        |    O365,GoldG    |                                                                |                                                                                                                    |
|  Grounding DINO-T   |  Swin-T  | Zero-shot |    48.5    |       48.4        | O365,GoldG,Cap4M | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 48.5(+1.8) |                   |       O365       |                           [config]()                           |                                                     [model]()                                                      |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.4(+2.3) |                   |    O365,GoldG    |                           [config]()                           |                                                     [model]()                                                      |
| Grounding DINO-T-V2 |  Swin-T  | Zero-shot | 50.5(+2.1) |                   | O365,GoldG,GRIT  |                           [config]()                           |                                                     [model]()                                                      |

## LVIS Results

|        Model        | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP  |  Pre-Train Data  |   Config   | Download  |
|:-------------------:|:-----------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------------:|:----------:|:---------:|
|  Grounding DINO-T   |    18.8     |    24.2     |    34.7     |    28.8    |    10.1    |    15.3    |    29.9    |    20.1    | O365,GoldG,Cap4M | [config]() | [model]() |
| Grounding DINO-T-V2 |    28.1     |    30.2     |    42.0     | 35.7(+6.9) |    17.1    |    22.4    |    36.5    | 27.0(+6.9) |    O365,GoldG    | [config]() | [model]() |
| Grounding DINO-T-V2 |    26.6     |    32.4     |    41.8     | 36.5(+7.7) |        |        |        |            | O365,GoldG,GRIT  | [config]() | [model]() |


## ODinW (Object Detection in the Wild) Results

Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets and tasks. However, it remains challenging to evaluate the transferablity of these models due to the lack of easy-to-use evaluation toolkits and public benchmarks. To tackle this, we build ELEVATER 1 , the first benchmark and toolkit for evaluating (pre-trained) language-augmented visual models. ELEVATER is composed of three components. (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to facilitate model evaluation on downstream tasks. (iii) Metrics. A variety of evaluation metrics are used to measure sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). ELEVATER is platform for Computer Vision in the Wild (CVinW), and is publicly released at https://computer-vision-in-the-wild.github.io/ELEVATER/

### Results and models of ODinW13

| Method                | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) |
|-----------------------|------------------------------------------|---------------------------------------|--------------------------------------------|
| AerialMaritimeDrone   | 0.173                                    | 0.133                                 | 0.155                                      |
| Aquarium              | 0.195                                    | 0.252                                 | 0.261                                      |
| CottontailRabbits     | 0.799                                    | 0.771                                 | 0.810                                      |
| EgoHands              | 0.608                                    | 0.499                                 | 0.537                                      |
| NorthAmericaMushrooms | 0.507                                    | 0.331                                 | 0.462                                      |
| Packages              | 0.687                                    | 0.707                                 | 0.687                                      |
| PascalVOC             | 0.563                                    | 0.565                                 | 0.580                                      |
| pistols               | 0.726                                    | 0.585                                 | 0.709                                      |
| pothole               | 0.215                                    | 0.136                                 | 0.285                                      |
| Raccoon               | 0.549                                    | 0.469                                 | 0.511                                      |
| ShellfishOpenImages   | 0.393                                    | 0.321                                 | 0.437                                      |
| thermalDogsAndPeople  | 0.657                                    | 0.556                                 | 0.603                                      |
| VehiclesOpenImages    | 0.613                                    | 0.566                                 | 0.603                                      |
| Average               | **0.514**                                | **0.453**                             | **0.511**                                  |

### Results and models of ODinW35

| Method                      | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) |
|-----------------------------|------------------------------------------|---------------------------------------|--------------------------------------------|
| AerialMaritimeDrone_large   | 0.173                                    | 0.133                                 | 0.155                                      |
| AerialMaritimeDrone_tiled   | 0.206                                    | 0.170                                 | 0.225                                      |
| AmericanSignLanguageLetters | 0.002                                    | 0.016                                 | 0.020                                      |
| Aquarium                    | 0.195                                    | 0.252                                 | 0.261                                      |
| BCCD                        | 0.161                                    | 0.069                                 | 0.118                                      |
| boggleBoards                | 0.000                                    | 0.002                                 | 0.001                                      |
| brackishUnderwater          | 0.021                                    | 0.033                                 | 0.021                                      |
| ChessPieces                 | 0.000                                    | 0.000                                 | 0.000                                      |
| CottontailRabbits           | 0.806                                    | 0.771                                 | 0.810                                      |
| dice                        | 0.004                                    | 0.002                                 | 0.005                                      |
| DroneControl                | 0.042                                    | 0.047                                 | 0.097                                      |
| EgoHands_generic            | 0.608                                    | 0.527                                 | 0.537                                      |
| EgoHands_specific           | 0.002                                    | 0.001                                 | 0.005                                      |
| HardHatWorkers              | 0.046                                    | 0.048                                 | 0.070                                      |
| MaskWearing                 | 0.004                                    | 0.009                                 | 0.004                                      |
| MountainDewCommercial       | 0.430                                    | 0.453                                 | 0.465                                      |
| NorthAmericaMushrooms       | 0.471                                    | 0.331                                 | 0.462                                      |
| openPoetryVision            | 0.000                                    | 0.001                                 | 0.000                                      |
| OxfordPets_by_breed         | 0.003                                    | 0.002                                 | 0.004                                      |
| OxfordPets_by_species       | 0.011                                    | 0.019                                 | 0.016                                      |
| PKLot                       | 0.001                                    | 0.004                                 | 0.002                                      |
| Packages                    | 0.695                                    | 0.707                                 | 0.687                                      |
| PascalVOC                   | 0.563                                    | 0.565                                 | 0.580                                      |
| pistols                     | 0.726                                    | 0.585                                 | 0.709                                      |
| plantdoc                    | 0.005                                    | 0.005                                 | 0.007                                      |
| pothole                     | 0.215                                    | 0.136                                 | 0.219                                      |
| Raccoons                    | 0.549                                    | 0.469                                 | 0.511                                      |
| selfdrivingCar              | 0.089                                    | 0.091                                 | 0.076                                      |
| ShellfishOpenImages         | 0.393                                    | 0.321                                 | 0.437                                      |
| ThermalCheetah              | 0.087                                    | 0.063                                 | 0.081                                      |
| thermalDogsAndPeople        | 0.657                                    | 0.556                                 | 0.603                                      |
| UnoCards                    | 0.006                                    | 0.012                                 | 0.010                                      |
| VehiclesOpenImages          | 0.613                                    | 0.566                                 | 0.603                                      |
| WildfireSmoke               | 0.134                                    | 0.106                                 | 0.154                                      |
| websiteScreenshots          | 0.012                                    | 0.02                                  | 0.016                                      |
| Average                     | **0.227**                                | **0.202**                             | **0.228**                                  |

## Referring Expression Comprehension Results

| Method                                  | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) |
|-----------------------------------------|------------------------------------------|---------------------------------------|--------------------------------------------|
| RefCOCO val @1,5,10                     | 50.77/89.45/94.86                        | 53.06/89.91/94.69                     | 53.4/90.3/95.5                             |
| RefCOCO testA @1,5,10                   | 57.45/91.29/95.62                        | 59.70/91.50/95.88                     | 58.8/91.70/96.2                            |
| RefCOCO testB @1,5,10                   | 44.97/86.54/92.88                        | 46.38/86.87/92.21                     | 46.8/87.7/93.3                             |
| RefCOCO+ val @1,5,10                    | 51.64/86.35/92.57                        | 53.11/87.00/92.79                     | 53.5/88.00/93.7                            |
| RefCOCO+ testA @1,5,10                  | 57.25/86.74/92.65                        | 58.94/87.34/92.91                     | 59.0/88.1/93.7                             |
| RefCOCO+ testB @1,5,10                  | 46.35/84.05/90.67                        | 47.92/84.31/91.04                     | 47.9/85.5/92.7                             |
| RefCOCOg val @1,5,10                    | 60.42/92.10/96.18                        | 61.23/92.61/96.14                     | 62.7/93.3/97.0                             |
| RefCOCOg test @1,5,10                   | 59.74/92.08/96.28                        | 61.13/93.26/96.72                     | 62.6/94.9/97.1                             |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 41.32/91.82                              | 39.76/84.65                           | 40.7/89.7                                  |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 27.23/90.24                              | 26.25/89.04                           | 26.0/91.9                                  |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 29.70/93.49                              | 31.31/84.79                           | 30.6/90.2                                  |

## Description Detection Dataset

```shell
pip install ddd-dataset
```

| Method                           | GroundingDINO-T <br/> (O365,GoldG,Cap4M) | GroundingDINO-T-V2 <br/> (O365,GoldG) | GroundingDINO-T-V2 <br/> (O365,GoldG)-20e | GroundingDINO-T-V2 <br/> (O365,GoldG,GRIT) |
|----------------------------------|------------------------------------------|---------------------------------------|-------------------------------------------|--------------------------------------------|
| FULL/short/middle/long/very long | 17.2/18.0/18.7/14.8/16.3                 | 15.6/17.3/16.7/14.3/13.1              | 16.3/17.2/17.2/15.1/15.1                  | 17.0/17.7/18.0/15.7/15.7                   |
| PRES/short/middle/long/very long | 17.8/18.3/19.2/15.2/17.3                 | 16.4/18.4/17.3/14.5/14.2              | 16.9/18.2/17.6/15.5/15.6                  | 17.9/19.0/18.3/16.5/17.5                   |
| ABS/short/middle/long/very long  | 15.4/17.1/16.4/13.6/14.9                 | 13.4/13.4/14.5/13.5/11.9              | 14.6/13.7/15.2/14.2/14.6                  | 14.5/13.1/16.7/13.6/13.3                   |

Note:

1. Intra-scenario mAP

