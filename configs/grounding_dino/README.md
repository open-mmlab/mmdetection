# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

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

## NOTE

Grounding DINO utilizes BERT as the language model, which requires access to https://huggingface.co/. If you encounter connection errors due to network access, you can download the required files on a computer with internet access and save them locally. Finally, modify the `lang_model_name` field in the config to the local path. Please refer to the following code:

```python
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("your path/bert-base-uncased")
model.save_pretrained("your path/bert-base-uncased")
tokenizer.save_pretrained("your path/bert-base-uncased")
```

## Inference

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

## COCO Results and Models

|       Model        | Backbone |   Style   |  COCO mAP  | Official COCO mAP |                  Pre-Train Data                  |                             Config                             |                                                                                                                                                                                                                                         Download                                                                                                                                                                                                                                          |
| :----------------: | :------: | :-------: | :--------: | :---------------: | :----------------------------------------------: | :------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Grounding DINO-T  |  Swin-T  | Zero-shot |    48.5    |       48.4        |                 O365,GoldG,Cap4M                 | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) |                                                                                                                                                                                    [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)                                                                                                                                                                                     |
|  Grounding DINO-T  |  Swin-T  | Finetune  | 58.1(+0.9) |       57.2        |                 O365,GoldG,Cap4M                 |   [config](grounding_dino_swin-t_finetune_16xb2_1x_coco.py)    | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth)                                                                                                \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544.log.json) |
|  Grounding DINO-B  |  Swin-B  | Zero-shot |    56.9    |       56.7        | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO |     [config](grounding_dino_swin-b_pretrain_mixeddata.py)      |                                                                                                                                                                                  [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth)                                                                                                                                                                                   |
|  Grounding DINO-B  |  Swin-B  | Finetune  |    59.7    |                   | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO |   [config](grounding_dino_swin-b_finetune_16xb2_1x_coco.py)    |                                               [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth)   \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201.log.json)                                                |
| Grounding DINO-R50 |   R50    |  Scratch  | 48.9(+0.8) |       48.1        |                                                  |      [config](grounding_dino_r50_scratch_8xb2_1x_coco.py)      |                                                                                          [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco/grounding_dino_r50_scratch_1x_coco-fe0002f2.pth)  \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco/20230922_114218.json)                                                                                           |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/groundingdino_to_mmdet.py). We have not retrained the model for the time being.
2. Finetune refers to fine-tuning on the COCO 2017 dataset. The R50 model is trained using 8 NVIDIA GeForce 3090 GPUs, while the remaining models are trained using 16 NVIDIA GeForce 3090 GPUs. The GPU memory usage is approximately 8.5GB.
3. Our performance is higher than the official model due to two reasons: we modified the initialization strategy and introduced a log scaler.

## LVIS Results

|      Model       | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP |                  Pre-Train Data                  |                               Config                                |                                                        Download                                                        |
| :--------------: | :---------: | :---------: | :---------: | :--------: | :--------: | :--------: | :--------: | :-------: | :----------------------------------------------: | :-----------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| Grounding DINO-T |    18.8     |    24.2     |    34.7     |    28.8    |    10.1    |    15.3    |    29.9    |   20.1    |                 O365,GoldG,Cap4M                 | [config](lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py) |   [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)   |
| Grounding DINO-B |    27.9     |    33.4     |    37.2     |    34.7    |    19.0    |    24.1    |    32.9    |   26.7    | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO | [config](lvis/grounding_dino_swin-b_pretrain_zeroshot_mini-lvis.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth) |

Note:

1. The above are zero-shot evaluation results.
2. The evaluation metric we used is LVIS FixAP. For specific details, please refer to [Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details](https://arxiv.org/pdf/2102.01066.pdf).

## ODinW (Object Detection in the Wild) Results

Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets and tasks. However, it remains challenging to evaluate the transferablity of these models due to the lack of easy-to-use evaluation toolkits and public benchmarks. To tackle this, we build ELEVATER 1 , the first benchmark and toolkit for evaluating (pre-trained) language-augmented visual models. ELEVATER is composed of three components. (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to facilitate model evaluation on downstream tasks. (iii) Metrics. A variety of evaluation metrics are used to measure sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). ELEVATER is platform for Computer Vision in the Wild (CVinW), and is publicly released at https://computer-vision-in-the-wild.github.io/ELEVATER/

### Results and models of ODinW13

| Method                | GLIP-T(A) | Official  | GLIP-T(B) | Official  | GLIP-T(C) | Official  | GroundingDINO-T | GroundingDINO-B |
| --------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------------- | --------------- |
| AerialMaritimeDrone   | 0.123     | 0.122     | 0.110     | 0.110     | 0.130     | 0.130     | 0.173           | 0.281           |
| Aquarium              | 0.175     | 0.174     | 0.173     | 0.169     | 0.191     | 0.190     | 0.195           | 0.445           |
| CottontailRabbits     | 0.686     | 0.686     | 0.688     | 0.688     | 0.744     | 0.744     | 0.799           | 0.808           |
| EgoHands              | 0.013     | 0.013     | 0.003     | 0.004     | 0.314     | 0.315     | 0.608           | 0.764           |
| NorthAmericaMushrooms | 0.502     | 0.502     | 0.367     | 0.367     | 0.297     | 0.296     | 0.507           | 0.675           |
| Packages              | 0.589     | 0.589     | 0.083     | 0.083     | 0.699     | 0.699     | 0.687           | 0.670           |
| PascalVOC             | 0.512     | 0.512     | 0.541     | 0.540     | 0.565     | 0.565     | 0.563           | 0.711           |
| pistols               | 0.339     | 0.339     | 0.502     | 0.501     | 0.503     | 0.504     | 0.726           | 0.771           |
| pothole               | 0.007     | 0.007     | 0.030     | 0.030     | 0.058     | 0.058     | 0.215           | 0.478           |
| Raccoon               | 0.075     | 0.074     | 0.285     | 0.288     | 0.241     | 0.244     | 0.549           | 0.541           |
| ShellfishOpenImages   | 0.253     | 0.253     | 0.337     | 0.338     | 0.300     | 0.302     | 0.393           | 0.650           |
| thermalDogsAndPeople  | 0.372     | 0.372     | 0.475     | 0.475     | 0.510     | 0.510     | 0.657           | 0.633           |
| VehiclesOpenImages    | 0.574     | 0.566     | 0.562     | 0.547     | 0.549     | 0.534     | 0.613           | 0.647           |
| Average               | **0.325** | **0.324** | **0.320** | **0.318** | **0.392** | **0.392** | **0.514**       | **0.621**       |

### Results and models of ODinW35

| Method                      | GLIP-T(A) | Official  | GLIP-T(B) | Official  | GLIP-T(C) | Official  | GroundingDINO-T | GroundingDINO-B |
| --------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------------- | --------------- |
| AerialMaritimeDrone_large   | 0.123     | 0.122     | 0.110     | 0.110     | 0.130     | 0.130     | 0.173           | 0.281           |
| AerialMaritimeDrone_tiled   | 0.174     | 0.174     | 0.172     | 0.172     | 0.172     | 0.172     | 0.206           | 0.364           |
| AmericanSignLanguageLetters | 0.001     | 0.001     | 0.003     | 0.003     | 0.009     | 0.009     | 0.002           | 0.096           |
| Aquarium                    | 0.175     | 0.175     | 0.173     | 0.171     | 0.192     | 0.182     | 0.195           | 0.445           |
| BCCD                        | 0.016     | 0.016     | 0.001     | 0.001     | 0.000     | 0.000     | 0.161           | 0.584           |
| boggleBoards                | 0.000     | 0.000     | 0.000     | 0.000     | 0.000     | 0.000     | 0.000           | 0.134           |
| brackishUnderwater          | 0.016     | 0..013    | 0.021     | 0.027     | 0.020     | 0.022     | 0.021           | 0.454           |
| ChessPieces                 | 0.001     | 0.001     | 0.000     | 0.000     | 0.001     | 0.001     | 0.000           | 0.000           |
| CottontailRabbits           | 0.710     | 0.709     | 0.683     | 0.683     | 0.752     | 0.752     | 0.806           | 0.797           |
| dice                        | 0.005     | 0.005     | 0.004     | 0.004     | 0.004     | 0.004     | 0.004           | 0.082           |
| DroneControl                | 0.016     | 0.017     | 0.006     | 0.008     | 0.005     | 0.007     | 0.042           | 0.638           |
| EgoHands_generic            | 0.009     | 0.010     | 0.005     | 0.006     | 0.510     | 0.508     | 0.608           | 0.764           |
| EgoHands_specific           | 0.001     | 0.001     | 0.004     | 0.006     | 0.003     | 0.004     | 0.002           | 0.687           |
| HardHatWorkers              | 0.029     | 0.029     | 0.023     | 0.023     | 0.033     | 0.033     | 0.046           | 0.439           |
| MaskWearing                 | 0.007     | 0.007     | 0.003     | 0.002     | 0.005     | 0.005     | 0.004           | 0.406           |
| MountainDewCommercial       | 0.218     | 0.227     | 0.199     | 0.197     | 0.478     | 0.463     | 0.430           | 0.580           |
| NorthAmericaMushrooms       | 0.502     | 0.502     | 0.450     | 0.450     | 0.497     | 0.497     | 0.471           | 0.501           |
| openPoetryVision            | 0.000     | 0.000     | 0.000     | 0.000     | 0.000     | 0.000     | 0.000           | 0.051           |
| OxfordPets_by_breed         | 0.001     | 0.002     | 0.002     | 0.004     | 0.001     | 0.002     | 0.003           | 0.799           |
| OxfordPets_by_species       | 0.016     | 0.011     | 0.012     | 0.009     | 0.013     | 0.009     | 0.011           | 0.872           |
| PKLot                       | 0.002     | 0.002     | 0.000     | 0.000     | 0.000     | 0.000     | 0.001           | 0.774           |
| Packages                    | 0.569     | 0.569     | 0.279     | 0.279     | 0.712     | 0.712     | 0.695           | 0.728           |
| PascalVOC                   | 0.512     | 0.512     | 0.541     | 0.540     | 0.565     | 0.565     | 0.563           | 0.711           |
| pistols                     | 0.339     | 0.339     | 0.502     | 0.501     | 0.503     | 0.504     | 0.726           | 0.771           |
| plantdoc                    | 0.002     | 0.002     | 0.007     | 0.007     | 0.009     | 0.009     | 0.005           | 0.376           |
| pothole                     | 0.007     | 0.010     | 0.024     | 0.025     | 0.085     | 0.101     | 0.215           | 0.478           |
| Raccoons                    | 0.075     | 0.074     | 0.285     | 0.288     | 0.241     | 0.244     | 0.549           | 0.541           |
| selfdrivingCar              | 0.071     | 0.072     | 0.074     | 0.074     | 0.081     | 0.080     | 0.089           | 0.318           |
| ShellfishOpenImages         | 0.253     | 0.253     | 0.337     | 0.338     | 0.300     | 0.302     | 0.393           | 0.650           |
| ThermalCheetah              | 0.028     | 0.028     | 0.000     | 0.000     | 0.028     | 0.028     | 0.087           | 0.290           |
| thermalDogsAndPeople        | 0.372     | 0.372     | 0.475     | 0.475     | 0.510     | 0.510     | 0.657           | 0.633           |
| UnoCards                    | 0.000     | 0.000     | 0.000     | 0.001     | 0.002     | 0.003     | 0.006           | 0.754           |
| VehiclesOpenImages          | 0.574     | 0.566     | 0.562     | 0.547     | 0.549     | 0.534     | 0.613           | 0.647           |
| WildfireSmoke               | 0.000     | 0.000     | 0.000     | 0.000     | 0.017     | 0.017     | 0.134           | 0.410           |
| websiteScreenshots          | 0.003     | 0.004     | 0.003     | 0.005     | 0.005     | 0.006     | 0.012           | 0.175           |
| Average                     | **0.134** | **0.134** | **0.138** | **0.138** | **0.179** | **0.178** | **0.227**       | **0.492**       |

## Flickr30k Results

|      Model       |  Pre-Train Data  | Val R@1 | Val R@5 | Val R@10 | Tesst R@1 | Test R@5 | Test R@10 |                          Config                           |                                                                                                                                                                                                                                         Download                                                                                                                                                                                                                                          |
| :--------------: | :--------------: | ------- | ------- | -------- | --------- | -------- | --------- | :-------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Grounding DINO-T | O365,GoldG,Cap4M | 87.8    | 96.6    | 98.0     | 88.1      | 96.9     | 98.2      | [config](grounding_dino_swin-t_finetune_16xb2_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth)                                                                                                \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544.log.json) |

Note:

1. `@1,5,10` refers to precision at the top 1, 5, and 10 positions in a predicted ranked list.
2. The pretraining data used by Grounding DINO-T is `O365,GoldG,Cap4M`, and the corresponding evaluation configuration is (grounding_dino_swin-t_pretrain_zeroshot_refcoco)\[refcoco/grounding_dino_swin-t_pretrain_zeroshot_refcoco.py\].

Test Command

```shell
cd mmdetection
bash tools/dist_test.sh configs/grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_zeroshot_flickr30k.py checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth 8
```

## Referring Expression Comprehension Results

| Method                                  | Grounding DINO-T <br/> (O365,GoldG,Cap4M) | Grounding DINO-B <br/> (COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO) |
| --------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------- |
| RefCOCO val @1,5,10                     | 50.77/89.45/94.86                         | 84.61/97.88/99.10                                                         |
| RefCOCO testA @1,5,10                   | 57.45/91.29/95.62                         | 88.65/98.89/99.63                                                         |
| RefCOCO testB @1,5,10                   | 44.97/86.54/92.88                         | 80.51/96.64/98.51                                                         |
| RefCOCO+ val @1,5,10                    | 51.64/86.35/92.57                         | 73.67/96.60/98.65                                                         |
| RefCOCO+ testA @1,5,10                  | 57.25/86.74/92.65                         | 82.19/97.92/99.09                                                         |
| RefCOCO+ testB @1,5,10                  | 46.35/84.05/90.67                         | 64.10/94.25/97.46                                                         |
| RefCOCOg val @1,5,10                    | 60.42/92.10/96.18                         | 78.33/97.28/98.57                                                         |
| RefCOCOg test @1,5,10                   | 59.74/92.08/96.28                         | 78.11/97.06/98.65                                                         |
| gRefCOCO val Pr@(F1=1, IoU≥0.5),N-acc   | 41.32/91.82                               | 46.18/81.44                                                               |
| gRefCOCO testA Pr@(F1=1, IoU≥0.5),N-acc | 27.23/90.24                               | 38.60/76.06                                                               |
| gRefCOCO testB Pr@(F1=1, IoU≥0.5),N-acc | 29.70/93.49                               | 35.87/80.58                                                               |

Note:

1. `@1,5,10` refers to precision at the top 1, 5, and 10 positions in a predicted ranked list.
2. `Pr@(F1=1, IoU≥0.5),N-acc` from the paper [GREC: Generalized Referring Expression Comprehension](https://arxiv.org/pdf/2308.16182.pdf)
3. The pretraining data used by Grounding DINO-T is `O365,GoldG,Cap4M`, and the corresponding evaluation configuration is (grounding_dino_swin-t_pretrain_zeroshot_refcoco)\[refcoco/grounding_dino_swin-t_pretrain_zeroshot_refcoco.py\].
4. The pretraining data used by Grounding DINO-B is `COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO`, and the corresponding evaluation configuration is (grounding_dino_swin-t_pretrain_zeroshot_refcoco)\[refcoco/grounding_dino_swin-b_pretrain_zeroshot_refcoco.py\].

Test Command

```shell
cd mmdetection
./tools/dist_test.sh configs/grounding_dino/refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth 8
./tools/dist_test.sh configs/grounding_dino/refcoco/grounding_dino_swin-b_pretrain_zeroshot_refexp.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth 8
```

## Description Detection Dataset

```shell
pip install ddd-dataset
```

| Method                           | mode     | Grounding DINO-T <br/> (O365,GoldG,Cap4M) | Grounding DINO-B <br/> (COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO) |
| -------------------------------- | -------- | ----------------------------------------- | ------------------------------------------------------------------------- |
| FULL/short/middle/long/very long | concat   | 17.2/18.0/18.7/14.8/16.3                  | 20.2/20.4/21.1/18.8/19.8                                                  |
| FULL/short/middle/long/very long | parallel | 22.3/28.2/24.8/19.1/13.9                  | 25.0/26.4/27.2/23.5/19.7                                                  |
| PRES/short/middle/long/very long | concat   | 17.8/18.3/19.2/15.2/17.3                  | 20.7/21.7/21.4/19.1/20.3                                                  |
| PRES/short/middle/long/very long | parallel | 21.0/27.0/22.8/17.5/12.5                  | 23.7/25.8/25.1/21.9/19.3                                                  |
| ABS/short/middle/long/very long  | concat   | 15.4/17.1/16.4/13.6/14.9                  | 18.6/16.1/19.7/18.1/19.1                                                  |
| ABS/short/middle/long/very long  | parallel | 26.0/32.0/33.0/23.6/15.5                  | 28.8/28.1/35.8/28.2/20.2                                                  |

Note:

1. Considering that the evaluation time for Inter-scenario is very long and the performance is low, it is temporarily not supported. The mentioned metrics are for Intra-scenario.
2. `concat` is the default inference mode for Grounding DINO, where it concatenates multiple sub-sentences with "." to form a single sentence for inference. On the other hand, "parallel" performs inference on each sub-sentence in a for-loop.

## Custom Dataset

To facilitate fine-tuning on custom datasets, we use a simple cat dataset as an example, as shown in the following steps.

### 1. Dataset Preparation

```shell
cd mmdetection
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d data/cat/
```

cat dataset is a single-category dataset with 144 images, which has been converted to coco format.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

### 2. Config Preparation

Due to the simplicity and small number of cat datasets, we use 8 cards to train 20 epochs, scale the learning rate accordingly, and do not train the language model, only the visual model.

The Details of the configuration can be found in [grounding_dino_swin-t_finetune_8xb2_20e_cat](grounding_dino_swin-t_finetune_8xb2_20e_cat.py)

### 3. Visualization and Evaluation

Due to the Grounding DINO is an open detection model, so it can be detected and evaluated even if it is not trained on the cat dataset.

The single image visualization is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/89173261-16f1-4fd9-ac63-8dc2dcda6616" alt="cat dataset"/>
</div>

The test dataset evaluation on single card is as follows:

```shell
python tools/test.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.931
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.903
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.907
```

### 4. Model Training and Visualization

```shell
./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py 8 --work-dir cat_work_dir
```

The model will be saved based on the best performance on the test set. The performance of the best model (at epoch 16) is as follows:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.905
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.923
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.905
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.927
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.937
```

We can find that after fine-tuning training, the training of the cat dataset is increased from 86.7 to 90.5.

If we do single image inference visualization again, the result is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights cat_work_dir/best_coco_bbox_mAP_epoch_16.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5a027b00-8adb-4283-a47b-2f7a0a2c96d4" alt="cat dataset"/>
</div>
