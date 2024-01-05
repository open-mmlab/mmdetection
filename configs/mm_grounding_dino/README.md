# MM Grounding DINO

> [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://arxiv.org/abs/2401.02361)

<!-- [ALGORITHM] -->

## Abstract

Grounding-DINO is a state-of-the-art open-set detection model that tackles multiple vision tasks including Open-Vocabulary Detection (OVD), Phrase Grounding (PG), and Referring Expression Comprehension (REC). Its effectiveness has led to its widespread adoption as a mainstream architecture for various downstream applications. However, despite its significance, the original Grounding-DINO model lacks comprehensive public technical details due to the unavailability of its training code. To bridge this gap, we present MM-Grounding-DINO, an open-source, comprehensive, and user-friendly baseline, which is built with the MMDetection toolbox. It adopts abundant vision datasets for pre-training and various detection and grounding datasets for fine-tuning. We give a comprehensive analysis of each reported result and detailed settings for reproduction. The extensive experiments on the benchmarks mentioned demonstrate that our MM-Grounding-DINO-Tiny outperforms the Grounding-DINO-Tiny baseline. We release all our models to the research community.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/4214e282-a553-4abf-b8a4-84ea566851c9"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/fb14d1ee-5469-44d2-b865-aac9850c429c"/>
</div>

## Dataset Preparation

Please refer to [dataset_prepare.md](dataset_prepare.md) or [中文版数据准备](dataset_prepare_zh-CN.md)

## Usage

Please refer to [usage.md](usage.md) or [中文版用法说明](usage_zh-CN.md)

## Zero-Shot COCO Results and Models

|   Model    | Backbone |   Style   |  COCO mAP  |    Pre-Train Data     |                                      Config                                      |                                                                                                                                                                                                                     Download                                                                                                                                                                                                                      |
| :--------: | :------: | :-------: | :--------: | :-------------------: | :------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  GDINO-T   |  Swin-T  | Zero-shot |    46.7    |         O365          |                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|  GDINO-T   |  Swin-T  | Zero-shot |    48.1    |      O365,GoldG       |                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|  GDINO-T   |  Swin-T  | Zero-shot |    48.4    |   O365,GoldG,Cap4M    | [config](../grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) |                                                                                                                                                                [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)                                                                                                                                                                 |
| MM-GDINO-T |  Swin-T  | Zero-shot | 48.5(+1.8) |         O365          |                [config](grounding_dino_swin-t_pretrain_obj365.py)                |                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| MM-GDINO-T |  Swin-T  | Zero-shot | 50.4(+2.3) |      O365,GoldG       |             [config](grounding_dino_swin-t_pretrain_obj365_goldg.py)             |                           [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602.log.json)                            |
| MM-GDINO-T |  Swin-T  | Zero-shot | 50.5(+2.1) |    O365,GoldG,GRIT    |         [config](grounding_dino_swin-t_pretrain_obj365_goldg_grit9m.py)          |             [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818.log.json)              |
| MM-GDINO-T |  Swin-T  | Zero-shot | 50.6(+2.2) |   O365,GoldG,V3Det    |          [config](grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py)          |             [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth)      \| [log](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741.log.json)             |
| MM-GDINO-T |  Swin-T  | Zero-shot | 50.4(+2.0) | O365,GoldG,GRIT,V3Det |      [config](grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py)       | [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth)  \| [log](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047.log.json) |

## Zero-Shot LVIS Results

|   Model    | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |    Pre-Train Data     |
| :--------: | :---------: | :---------: | :---------: | :---------: | :--------: | :--------: | :--------: | :---------: | :-------------------: |
|  GDINO-T   |    18.8     |    24.2     |    34.7     |    28.8     |    10.1    |    15.3    |    29.9    |    20.1     |   O365,GoldG,Cap4M    |
| MM-GDINO-T |    28.1     |    30.2     |    42.0     | 35.7(+6.9)  |    17.1    |    22.4    |    36.5    | 27.0(+6.9)  |      O365,GoldG       |
| MM-GDINO-T |    26.6     |    32.4     |    41.8     | 36.5(+7.7)  |    17.3    |    22.6    |    36.4    | 27.1(+7.0)  |    O365,GoldG,GRIT    |
| MM-GDINO-T |    33.0     |    36.0     |    45.9     | 40.5(+11.7) |    21.5    |    25.5    |    40.2    | 30.6(+10.5) |   O365,GoldG,V3Det    |
| MM-GDINO-T |    34.2     |    37.4     |    46.2     | 41.4(+12.6) |    23.6    |    27.6    |    40.5    | 31.9(+11.8) | O365,GoldG,GRIT,V3Det |

- The MM-GDINO-T config file is [mini-lvis](lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py) and [lvis 1.0](lvis/grounding_dino_swin-t_pretrain_zeroshot_lvis.py)

## Zero-Shot ODinW (Object Detection in the Wild) Results

### Results and models of ODinW13

| Method                | GDINO-T <br/> (O365,GoldG,Cap4M) | MM-GDINO-T <br/> (O365,GoldG) | MM-GDINO-T <br/> (O365,GoldG,GRIT) | MM-GDINO-T <br/> (O365,GoldG,V3Det) | MM-GDINO-T <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------- | -------------------------------- | ----------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------- |
| AerialMaritimeDrone   | 0.173                            | 0.133                         | 0.155                              | 0.177                               | 0.151                                    |
| Aquarium              | 0.195                            | 0.252                         | 0.261                              | 0.266                               | 0.283                                    |
| CottontailRabbits     | 0.799                            | 0.771                         | 0.810                              | 0.778                               | 0.786                                    |
| EgoHands              | 0.608                            | 0.499                         | 0.537                              | 0.506                               | 0.519                                    |
| NorthAmericaMushrooms | 0.507                            | 0.331                         | 0.462                              | 0.669                               | 0.767                                    |
| Packages              | 0.687                            | 0.707                         | 0.687                              | 0.710                               | 0.706                                    |
| PascalVOC             | 0.563                            | 0.565                         | 0.580                              | 0.556                               | 0.566                                    |
| pistols               | 0.726                            | 0.585                         | 0.709                              | 0.671                               | 0.729                                    |
| pothole               | 0.215                            | 0.136                         | 0.285                              | 0.199                               | 0.243                                    |
| Raccoon               | 0.549                            | 0.469                         | 0.511                              | 0.553                               | 0.535                                    |
| ShellfishOpenImages   | 0.393                            | 0.321                         | 0.437                              | 0.519                               | 0.488                                    |
| thermalDogsAndPeople  | 0.657                            | 0.556                         | 0.603                              | 0.493                               | 0.542                                    |
| VehiclesOpenImages    | 0.613                            | 0.566                         | 0.603                              | 0.614                               | 0.615                                    |
| Average               | **0.514**                        | **0.453**                     | **0.511**                          | **0.516**                           | **0.533**                                |

- The MM-GDINO-T config file is [odinw13](odinw/grounding_dino_swin-t_pretrain_odinw13.py)

### Results and models of ODinW35

| Method                      | GDINO-T <br/> (O365,GoldG,Cap4M) | MM-GDINO-T <br/> (O365,GoldG) | MM-GDINO-T <br/> (O365,GoldG,GRIT) | MM-GDINO-T <br/> (O365,GoldG,V3Det) | MM-GDINO-T <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------------- | -------------------------------- | ----------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------- |
| AerialMaritimeDrone_large   | 0.173                            | 0.133                         | 0.155                              | 0.177                               | 0.151                                    |
| AerialMaritimeDrone_tiled   | 0.206                            | 0.170                         | 0.225                              | 0.184                               | 0.206                                    |
| AmericanSignLanguageLetters | 0.002                            | 0.016                         | 0.020                              | 0.011                               | 0.007                                    |
| Aquarium                    | 0.195                            | 0.252                         | 0.261                              | 0.266                               | 0.283                                    |
| BCCD                        | 0.161                            | 0.069                         | 0.118                              | 0.083                               | 0.077                                    |
| boggleBoards                | 0.000                            | 0.002                         | 0.001                              | 0.001                               | 0.002                                    |
| brackishUnderwater          | 0.021                            | 0.033                         | 0.021                              | 0.025                               | 0.025                                    |
| ChessPieces                 | 0.000                            | 0.000                         | 0.000                              | 0.000                               | 0.000                                    |
| CottontailRabbits           | 0.806                            | 0.771                         | 0.810                              | 0.778                               | 0.786                                    |
| dice                        | 0.004                            | 0.002                         | 0.005                              | 0.001                               | 0.001                                    |
| DroneControl                | 0.042                            | 0.047                         | 0.097                              | 0.088                               | 0.074                                    |
| EgoHands_generic            | 0.608                            | 0.527                         | 0.537                              | 0.506                               | 0.519                                    |
| EgoHands_specific           | 0.002                            | 0.001                         | 0.005                              | 0.007                               | 0.003                                    |
| HardHatWorkers              | 0.046                            | 0.048                         | 0.070                              | 0.070                               | 0.108                                    |
| MaskWearing                 | 0.004                            | 0.009                         | 0.004                              | 0.011                               | 0.009                                    |
| MountainDewCommercial       | 0.430                            | 0.453                         | 0.465                              | 0.194                               | 0.430                                    |
| NorthAmericaMushrooms       | 0.471                            | 0.331                         | 0.462                              | 0.669                               | 0.767                                    |
| openPoetryVision            | 0.000                            | 0.001                         | 0.000                              | 0.000                               | 0.000                                    |
| OxfordPets_by_breed         | 0.003                            | 0.002                         | 0.004                              | 0.006                               | 0.004                                    |
| OxfordPets_by_species       | 0.011                            | 0.019                         | 0.016                              | 0.020                               | 0.015                                    |
| PKLot                       | 0.001                            | 0.004                         | 0.002                              | 0.008                               | 0.007                                    |
| Packages                    | 0.695                            | 0.707                         | 0.687                              | 0.710                               | 0.706                                    |
| PascalVOC                   | 0.563                            | 0.565                         | 0.580                              | 0.566                               | 0.566                                    |
| pistols                     | 0.726                            | 0.585                         | 0.709                              | 0.671                               | 0.729                                    |
| plantdoc                    | 0.005                            | 0.005                         | 0.007                              | 0.008                               | 0.011                                    |
| pothole                     | 0.215                            | 0.136                         | 0.219                              | 0.077                               | 0.168                                    |
| Raccoons                    | 0.549                            | 0.469                         | 0.511                              | 0.553                               | 0.535                                    |
| selfdrivingCar              | 0.089                            | 0.091                         | 0.076                              | 0.094                               | 0.083                                    |
| ShellfishOpenImages         | 0.393                            | 0.321                         | 0.437                              | 0.519                               | 0.488                                    |
| ThermalCheetah              | 0.087                            | 0.063                         | 0.081                              | 0.030                               | 0.045                                    |
| thermalDogsAndPeople        | 0.657                            | 0.556                         | 0.603                              | 0.493                               | 0.543                                    |
| UnoCards                    | 0.006                            | 0.012                         | 0.010                              | 0.009                               | 0.005                                    |
| VehiclesOpenImages          | 0.613                            | 0.566                         | 0.603                              | 0.614                               | 0.615                                    |
| WildfireSmoke               | 0.134                            | 0.106                         | 0.154                              | 0.042                               | 0.127                                    |
| websiteScreenshots          | 0.012                            | 0.02                          | 0.016                              | 0.016                               | 0.016                                    |
| Average                     | **0.227**                        | **0.202**                     | **0.228**                          | **0.214**                           | **0.284**                                |

- The MM-GDINO-T config file is [odinw35](odinw/grounding_dino_swin-t_pretrain_odinw35.py)

## Zero-Shot Referring Expression Comprehension Results

| Method                 | GDINO-T <br/> (O365,GoldG,Cap4M) | MM-GDINO-T <br/> (O365,GoldG) | MM-GDINO-T <br/> (O365,GoldG,GRIT) | MM-GDINO-T <br/> (O365,GoldG,V3Det) | MM-GDINO-T <br/> (O365,GoldG,GRIT,V3Det) |
| ---------------------- | -------------------------------- | ----------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------- |
| RefCOCO val @1,5,10    | 50.8/89.5/94.9                   | 53.1/89.9/94.7                | 53.4/90.3/95.5                     | 52.1/89.8/95.0                      | 53.1/89.7/95.1                           |
| RefCOCO testA @1,5,10  | 57.4/91.3/95.6                   | 59.7/91.5/95.9                | 58.8/91.70/96.2                    | 58.4/86.8/95.6                      | 59.1/91.0/95.5                           |
| RefCOCO testB @1,5,10  | 45.0/86.5/92.9                   | 46.4/86.9/92.2                | 46.8/87.7/93.3                     | 45.4/86.2/92.6                      | 46.8/87.8/93.6                           |
| RefCOCO+ val @1,5,10   | 51.6/86.4/92.6                   | 53.1/87.0/92.8                | 53.5/88.0/93.7                     | 52.5/86.8/93.2                      | 52.7/87.7/93.5                           |
| RefCOCO+ testA @1,5,10 | 57.3/86.7/92.7                   | 58.9/87.3/92.9                | 59.0/88.1/93.7                     | 58.1/86.7/93.5                      | 58.7/87.2/93.1                           |
| RefCOCO+ testB @1,5,10 | 46.4/84.1/90.7                   | 47.9/84.3/91.0                | 47.9/85.5/92.7                     | 46.9/83.7/91.5                      | 48.4/85.8/92.1                           |
| RefCOCOg val @1,5,10   | 60.4/92.1/96.2                   | 61.2/92.6/96.1                | 62.7/93.3/97.0                     | 61.7/92.9/96.6                      | 62.9/93.3/97.2                           |
| RefCOCOg test @1,5,10  | 59.7/92.1/96.3                   | 61.1/93.3/96.7                | 62.6/94.9/97.1                     | 61.0/93.1/96.8                      | 62.9/93.9/97.4                           |

| Method                                  | thresh_score | GDINO-T <br/> (O365,GoldG,Cap4M) | MM-GDINO-T <br/> (O365,GoldG) | MM-GDINO-T <br/> (O365,GoldG,GRIT) | MM-GDINO-T <br/> (O365,GoldG,V3Det) | MM-GDINO-T <br/> (O365,GoldG,GRIT,V3Det) |
| --------------------------------------- | ------------ | -------------------------------- | ----------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------- |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 0.5          | 39.3/70.4                        |                               |                                    |                                     | 39.4/67.5                                |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 0.6          | 40.5/83.8                        |                               |                                    |                                     | 40.6/83.1                                |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 0.7          | 41.3/91.8                        | 39.8/84.7                     | 40.7/89.7                          | 40.3/88.8                           | 41.0/91.3                                |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 0.8          | 41.5/96.8                        |                               |                                    |                                     | 41.1/96.4                                |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 0.5          | 31.9/70.4                        |                               |                                    |                                     | 33.1/69.5                                |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 0.6          | 29.3/82.9                        |                               |                                    |                                     | 29.2/84.3                                |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 0.7          | 27.2/90.2                        | 26.3/89.0                     | 26.0/91.9                          | 25.4/91.8                           | 26.1/93.0                                |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 0.8          | 25.1/96.3                        |                               |                                    |                                     | 23.8/97.2                                |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 0.5          | 30.9/72.5                        |                               |                                    |                                     | 33.0/69.6                                |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 0.6          | 30.0/86.1                        |                               |                                    |                                     | 31.6/96.7                                |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 0.7          | 29.7/93.5                        | 31.3/84.8                     | 30.6/90.2                          | 30.7/89.9                           | 30.4/92.3                                |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 0.8          | 29.1/97.4                        |                               |                                    |                                     | 29.5/84.2                                |

- The MM-GDINO-T config file is [here](refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp.py)

## Zero-Shot Description Detection Dataset(DOD)

```shell
pip install ddd-dataset
```

| Method                           | mode     | GDINO-T <br/> (O365,GoldG,Cap4M) | MM-GDINO-T <br/> (O365,GoldG) | MM-GDINO-T <br/> (O365,GoldG,GRIT) | MM-GDINO-T <br/> (O365,GoldG,V3Det) | MM-GDINO-T <br/> (O365,GoldG,GRIT,V3Det) |
| -------------------------------- | -------- | -------------------------------- | ----------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------- |
| FULL/short/middle/long/very long | concat   | 17.2/18.0/18.7/14.8/16.3         | 15.6/17.3/16.7/14.3/13.1      | 17.0/17.7/18.0/15.7/15.7           | 16.2/17.4/16.8/14.9/15.4            | 17.5/23.4/18.3/14.7/13.8                 |
| FULL/short/middle/long/very long | parallel | 22.3/28.2/24.8/19.1/13.9         | 21.7/24.7/24.0/20.2/13.7      | 22.5/25.6/25.1/20.5/14.9           | 22.3/25.6/24.5/20.6/14.7            | 22.9/28.1/25.4/20.4/14.4                 |
| PRES/short/middle/long/very long | concat   | 17.8/18.3/19.2/15.2/17.3         | 16.4/18.4/17.3/14.5/14.2      | 17.9/19.0/18.3/16.5/17.5           | 16.6/18.8/17.1/15.1/15.0            | 18.0/23.7/18.6/15.4/13.3                 |
| PRES/short/middle/long/very long | parallel | 21.0/27.0/22.8/17.5/12.5         | 21.3/25.5/22.8/19.2/12.9      | 21.5/25.2/23.0/19.0/15.0           | 21.6/25.7/23.0/19.5/14.8            | 21.9/27.4/23.2/19.1/14.2                 |
| ABS/short/middle/long/very long  | concat   | 15.4/17.1/16.4/13.6/14.9         | 13.4/13.4/14.5/13.5/11.9      | 14.5/13.1/16.7/13.6/13.3           | 14.8/12.5/15.6/14.3/15.8            | 15.9/22.2/17.1/12.5/14.4                 |
| ABS/short/middle/long/very long  | parallel | 26.0/32.0/33.0/23.6/15.5         | 22.8/22.2/28.7/22.9/14.7      | 25.6/26.8/33.9/24.5/14.7           | 24.1/24.9/30.7/23.8/14.7            | 26.0/30.3/34.1/23.9/14.6                 |

Note:

1. Considering that the evaluation time for Inter-scenario is very long and the performance is low, it is temporarily not supported. The mentioned metrics are for Intra-scenario.
2. `concat` is the default inference mode for Grounding DINO, where it concatenates multiple sub-sentences with "." to form a single sentence for inference. On the other hand, "parallel" performs inference on each sub-sentence in a for-loop.
3. The MM-GDINO-T config file is [concat_dod](dod/grounding_dino_swin-t_pretrain_zeroshot_concat_dod.py) and [parallel_dod](dod/grounding_dino_swin-t_pretrain_zeroshot_parallel_dod.py)

## Pretrain Flickr30k Results

|   Model    |    Pre-Train Data     | Val R@1 | Val R@5 | Val R@10 | Test R@1 | Test R@5 | Test R@10 |
| :--------: | :-------------------: | ------- | ------- | -------- | -------- | -------- | --------- |
|   GLIP-T   |      O365,GoldG       | 84.9    | 94.9    | 96.3     | 85.6     | 95.4     | 96.7      |
|   GLIP-T   |  O365,GoldG,CC3M,SBU  | 85.3    | 95.5    | 96.9     | 86.0     | 95.9     | 97.2      |
|  GDINO-T   |   O365,GoldG,Cap4M    | 87.8    | 96.6    | 98.0     | 88.1     | 96.9     | 98.2      |
| MM-GDINO-T |      O365,GoldG       | 85.5    | 95.6    | 97.2     | 86.2     | 95.7     | 97.4      |
| MM-GDINO-T |    O365,GoldG,GRIT    | 86.7    | 95.8    | 97.6     | 87.0     | 96.2     | 97.7      |
| MM-GDINO-T |   O365,GoldG,V3Det    | 85.9    | 95.7    | 97.4     | 86.3     | 95.7     | 97.4      |
| MM-GDINO-T | O365,GoldG,GRIT,V3Det | 86.7    | 96.0    | 97.6     | 87.2     | 96.2     | 97.7      |

Note:

1. `@1,5,10` refers to precision at the top 1, 5, and 10 positions in a predicted ranked list.
2. The MM-GDINO-T config file is [here](flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py)

## Validating the generalization of a pre-trained model through fine-tuning

### RTTS

|    Architecture     | Backbone | Lr schd | box AP   |
| :-----------------: | :------: | ------- | -------- |
|    Faster R-CNN     |   R-50   | 1x      | 48.1     |
|    Cascade R-CNN    |   R-50   | 1x      | 50.8     |
|        ATSS         |   R-50   | 1x      | 48.2     |
|        TOOD         |   R-50   | 1X      | 50.8     |
| MM-GDINO(zero-shot) |  Swin-T  |         | 49.8     |
|      MM-GDINO       |  Swin-T  | 1x      | **69.1** |

- The reference metrics come from https://github.com/BIGWangYuDong/lqit/tree/main/configs/detection/rtts_dataset
- The MM-GDINO-T config file is [here](rtts/grounding_dino_swin-t_finetune_8xb4_1x_rtts.py)

### RUOD

|    Architecture     | Backbone | Lr schd | box AP   |
| :-----------------: | :------: | ------- | -------- |
|    Faster R-CNN     |   R-50   | 1x      | 52.4     |
|    Cascade R-CNN    |   R-50   | 1x      | 55.3     |
|        ATSS         |   R-50   | 1x      | 55.7     |
|        TOOD         |   R-50   | 1X      | 57.4     |
| MM-GDINO(zero-shot) |  Swin-T  |         | 29.8     |
|      MM-GDINO       |  Swin-T  | 1x      | **65.5** |

- The reference metrics come from https://github.com/BIGWangYuDong/lqit/tree/main/configs/detection/ruod_dataset
- The MM-GDINO-T config file is [here](ruod/grounding_dino_swin-t_finetune_8xb4_1x_ruod.py)

### Brain Tumor

| Architecture  | Backbone | Lr schd | box AP |
| :-----------: | :------: | ------- | ------ |
| Faster R-CNN  |   R-50   | 50e     | 43.5   |
| Cascade R-CNN |   R-50   | 50e     | 46.2   |
|     DINO      |   R-50   | 50e     | 46.4   |
| Cascade-DINO  |   R-50   | 50e     | 48.6   |
|   MM-GDINO    |  Swin-T  | 50e     | 47.5   |

- The reference metrics come from https://arxiv.org/abs/2307.11035
- The MM-GDINO-T config file is [here](brain_tumor/grounding_dino_swin-t_finetune_8xb4_50e_brain_tumor.py)

### Cityscapes

|    Architecture     | Backbone | Lr schd | box AP   |
| :-----------------: | :------: | ------- | -------- |
|    Faster R-CNN     |   R-50   | 50e     | 30.1     |
|    Cascade R-CNN    |   R-50   | 50e     | 31.8     |
|        DINO         |   R-50   | 50e     | 34.5     |
|    Cascade-DINO     |   R-50   | 50e     | 34.8     |
| MM-GDINO(zero-shot) |  Swin-T  |         | 34.2     |
|      MM-GDINO       |  Swin-T  | 50e     | **51.5** |

- The reference metrics come from https://arxiv.org/abs/2307.11035
- The MM-GDINO-T config file is [here](cityscapes/grounding_dino_swin-t_finetune_8xb4_50e_cityscapes.py)

### People in Painting

|    Architecture     | Backbone | Lr schd | box AP   |
| :-----------------: | :------: | ------- | -------- |
|    Faster R-CNN     |   R-50   | 50e     | 17.0     |
|    Cascade R-CNN    |   R-50   | 50e     | 18.0     |
|        DINO         |   R-50   | 50e     | 12.0     |
|    Cascade-DINO     |   R-50   | 50e     | 13.4     |
| MM-GDINO(zero-shot) |  Swin-T  |         | 23.1     |
|      MM-GDINO       |  Swin-T  | 50e     | **38.9** |

- The reference metrics come from https://arxiv.org/abs/2307.11035
- The MM-GDINO-T config file is [here](people_in_painting/grounding_dino_swin-t_finetune_8xb4_50e_people_in_painting.py)

### COCO

**(1) Closed-set performance**

|    Architecture     | Backbone | Lr schd | box AP |
| :-----------------: | :------: | ------- | ------ |
|    Faster R-CNN     |   R-50   | 1x      | 37.4   |
|    Cascade R-CNN    |   R-50   | 1x      | 40.3   |
|        ATSS         |   R-50   | 1x      | 39.4   |
|        TOOD         |   R-50   | 1X      | 42.4   |
|        DINO         |   R-50   | 1X      | 50.1   |
|   GLIP(zero-shot)   |  Swin-T  |         | 46.6   |
|  GDINO(zero-shot)   |  Swin-T  |         | 48.5   |
| MM-GDINO(zero-shot) |  Swin-T  |         | 50.4   |
|        GLIP         |  Swin-T  | 1x      | 55.4   |
|        GDINO        |  Swin-T  | 1x      | 58.1   |
|      MM-GDINO       |  Swin-T  | 1x      | 58.2   |

- The MM-GDINO-T config file is [here](coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py)

**(2) Open-set continuing pretraining performance**

|    Architecture     | Backbone | Lr schd | box AP |
| :-----------------: | :------: | :-----: | :----: |
|   GLIP(zero-shot)   |  Swin-T  |         |  46.7  |
|  GDINO(zero-shot)   |  Swin-T  |         |  48.5  |
| MM-GDINO(zero-shot) |  Swin-T  |         |  50.4  |
|      MM-GDINO       |  Swin-T  |   1x    |  54.7  |

- The MM-GDINO-T config file is [here](coco/grounding_dino_swin-t_finetune_16xb4_1x_sft_coco.py)
- Due to the small size of the COCO dataset, continuing pretraining solely on COCO can easily lead to overfitting. The results shown above are from the third epoch. I do not recommend you train using this approach.

**(3) Open vocabulary performance**

|    Architecture     | Backbone | Lr schd | box AP | Base box AP | Novel box AP | box AP@50 | Base box AP@50 | Novel box AP@50 |
| :-----------------: | :------: | :-----: | :----: | :---------: | :----------: | :-------: | :------------: | :-------------: |
| MM-GDINO(zero-shot) |  Swin-T  |         |  51.1  |    48.4     |     58.9     |   66.7    |      64.0      |      74.2       |
|      MM-GDINO       |  Swin-T  |   1x    |  57.2  |    56.1     |     60.4     |   73.6    |      73.0      |      75.3       |

- The MM-GDINO-T config file is [here](coco/grounding_dino_swin-t_finetune_16xb4_1x_coco_48_17.py)

### LVIS 1.0

**(1) Open-set continuing pretraining performance**

|    Architecture     | Backbone | Lr schd | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP |
| :-----------------: | :------: | :-----: | :---------: | :---------: | :---------: | :--------: | :--------: | :--------: | :--------: | :-------: |
|   GLIP(zero-shot)   |  Swin-T  |         |    18.1     |    21.2     |    33.1     |    26.7    |    10.8    |    14.7    |    29.0    |   19.6    |
|  GDINO(zero-shot)   |  Swin-T  |         |    18.8     |    24.2     |    34.7     |    28.8    |    10.1    |    15.3    |    29.9    |   20.1    |
| MM-GDINO(zero-shot) |  Swin-T  |         |    34.2     |    37.4     |    46.2     |    41.4    |    23.6    |    27.6    |    40.5    |   31.9    |
|      MM-GDINO       |  Swin-T  |   1x    |    50.7     |    58.8     |    60.1     |    58.7    |    45.2    |    50.2    |    56.1    |   51.7    |

- The MM-GDINO-T config file is [here](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis.py)

**(2) Open vocabulary performance**

|    Architecture     | Backbone | Lr schd | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP |
| :-----------------: | :------: | :-----: | :---------: | :---------: | :---------: | :--------: |
| MM-GDINO(zero-shot) |  Swin-T  |         |    34.2     |    37.4     |    46.2     |    41.4    |
|      MM-GDINO       |  Swin-T  |   1x    |    43.2     |    57.4     |    59.3     |    57.1    |

- The MM-GDINO-T config file is [here](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py)

### RefEXP

#### RefCOCO

|    Architecture     | Backbone | Lr schd | val @1 | val @5 | val @10 | testA @1 | testA @5 | testA @10 | testB @1 | testB @5 | testB @10 |
| :-----------------: | :------: | :-----: | :----: | :----: | :-----: | :------: | :------: | :-------: | :------: | :------: | :-------: |
|  GDINO(zero-shot)   |  Swin-T  |         |  50.8  |  89.5  |  94.9   |   57.5   |   91.3   |   95.6    |   45.0   |   86.5   |   92.9    |
| MM-GDINO(zero-shot) |  Swin-T  |         |  53.1  |  89.7  |  95.1   |   59.1   |   91.0   |   95.5    |   46.8   |   87.8   |   93.6    |
|        GDINO        |  Swin-T  |   UNK   |  89.2  |        |         |   91.9   |          |           |   86.0   |          |           |
|      MM-GDINO       |  Swin-T  |   5e    |  89.5  |  98.6  |  99.4   |   91.4   |   99.2   |   99.8    |   86.6   |   97.9   |   99.1    |

- The MM-GDINO-T config file is [here](refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py)

#### RefCOCO+

|    Architecture     | Backbone | Lr schd | val @1 | val @5 | val @10 | testA @1 | testA @5 | testA @10 | testB @1 | testB @5 | testB @10 |
| :-----------------: | :------: | :-----: | :----: | :----: | :-----: | :------: | :------: | :-------: | :------: | :------: | :-------: |
|  GDINO(zero-shot)   |  Swin-T  |         |  51.6  |  86.4  |  92.6   |   57.3   |   86.7   |   92.7    |   46.4   |   84.1   |   90.7    |
| MM-GDINO(zero-shot) |  Swin-T  |         |  52.7  |  87.7  |  93.5   |   58.7   |   87.2   |   93.1    |   48.4   |   85.8   |   92.1    |
|        GDINO        |  Swin-T  |   UNK   |  81.1  |        |         |   87.4   |          |           |   74.7   |          |           |
|      MM-GDINO       |  Swin-T  |   5e    |  82.1  |  97.8  |  99.2   |   87.5   |   99.2   |   99.7    |   74.0   |   96.3   |   96.4    |

- The MM-GDINO-T config file is [here](refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco_plus.py)

#### RefCOCOg

|    Architecture     | Backbone | Lr schd | val @1 | val @5 | val @10 | test @1 | test @5 | test @10 |
| :-----------------: | :------: | :-----: | :----: | :----: | :-----: | :-----: | :-----: | :------: |
|  GDINO(zero-shot)   |  Swin-T  |         |  60.4  |  92.1  |  96.2   |  59.7   |  92.1   |   96.3   |
| MM-GDINO(zero-shot) |  Swin-T  |         |  62.9  |  93.3  |  97.2   |  62.9   |  93.9   |   97.4   |
|        GDINO        |  Swin-T  |   UNK   |  84.2  |        |         |  84.9   |         |          |
|      MM-GDINO       |  Swin-T  |   5e    |  85.5  |  98.4  |  99.4   |  85.8   |  98.6   |   99.4   |

- The MM-GDINO-T config file is [here](refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcocog.py)

#### gRefCOCO

|    Architecture     | Backbone | Lr schd | val Pr@(F1=1, IoU≥0.5) | val N-acc | testA Pr@(F1=1, IoU≥0.5) | testA N-acc | testB Pr@(F1=1, IoU≥0.5) | testB N-acc |
| :-----------------: | :------: | :-----: | :--------------------: | :-------: | :----------------------: | :---------: | :----------------------: | :---------: |
|  GDINO(zero-shot)   |  Swin-T  |         |          41.3          |   91.8    |           27.2           |    90.2     |           29.7           |    93.5     |
| MM-GDINO(zero-shot) |  Swin-T  |         |          41.0          |   91.3    |           26.1           |    93.0     |           30.4           |    92.3     |
|      MM-GDINO       |  Swin-T  |   5e    |          45.1          |   64.7    |           42.5           |    65.5     |           40.3           |    63.2     |

- The MM-GDINO-T config file is [here](refcoco/grounding_dino_swin-t_finetune_8xb4_5e_grefcoco.py)
