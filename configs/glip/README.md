# GLIP: Grounded Language-Image Pre-training

> [GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

```shell
cd $MMDETROOT

wget https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

python demo/image_demo.py demo/demo.jpg \
configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
--weights glip_tiny_a_mmdet-b3654169.pth \
--texts 'bench. car'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7b450d96-81ac-462a-92bc-0d4ae7b8721c" width="40%"/>
</div>

## NOTE

GLIP utilizes BERT as the language model, which requires access to https://huggingface.co/. If you encounter connection errors due to network access, you can download the required files on a computer with internet access and save them locally. Finally, modify the `lang_model_name` field in the config to the local path. Please refer to the following code:

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

## COCO Results and Models

|   Model    | Zero-shot or Finetune | COCO mAP | Official COCO mAP |       Pre-Train Data       |                                 Config                                  |                                                                                                                                                                                                   Download                                                                                                                                                                                                    |
| :--------: | :-------------------: | :------: | ----------------: | :------------------------: | :---------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GLIP-T (A) |       Zero-shot       |   43.0   |              42.9 |            O365            |       [config](glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth)                                                                                                                                                          |
| GLIP-T (A) |       Finetune        |   53.3   |              52.9 |            O365            |   [config](glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419.log.json) |
| GLIP-T (B) |       Zero-shot       |   44.9   |              44.9 |            O365            |       [config](glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth)                                                                                                                                                          |
| GLIP-T (B) |       Finetune        |   54.1   |              53.8 |            O365            |   [config](glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230916_163538-650323ba.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230916_163538.log.json) |
| GLIP-T (C) |       Zero-shot       |   46.7   |              46.7 |         O365,GoldG         |    [config](glip_atss_swin-t_c_fpn_dyhead_pretrain_obj365-goldg.py)     |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth)                                                                                                                                                          |
| GLIP-T (C) |       Finetune        |   55.2   |              55.1 |         O365,GoldG         |   [config](glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_182935-4ba3fc3b.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_182935.log.json) |
|   GLIP-T   |       Zero-shot       |   46.6   |              46.6 |    O365,GoldG,CC3M,SBU     | [config](glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub.py) |                                                                                                                                                          [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth)                                                                                                                                                           |
|   GLIP-T   |       Finetune        |   55.4   |              55.2 |    O365,GoldG,CC3M,SBU     |    [config](glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410-ba97be24.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410.log.json)     |
|   GLIP-L   |       Zero-shot       |   51.3   |              51.4 | FourODs,GoldG,CC3M+12M,SBU |       [config](glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py)       |                                                                                                                                                            [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth)                                                                                                                                                            |
|   GLIP-L   |       Finetune        |   59.4   |                   | FourODs,GoldG,CC3M+12M,SBU |    [config](glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800.log.json)     |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/glip_to_mmdet.py). We have not retrained the model for the time being.
2. Finetune refers to fine-tuning on the COCO 2017 dataset. The L model is trained using 16 A100 GPUs, while the remaining models are trained using 16 NVIDIA GeForce 3090 GPUs.
3. Taking the GLIP-T(A) model as an example, I trained it twice using the official code, and the fine-tuning mAP were 52.5 and 52.6. Therefore, the mAP we achieved in our reproduction is higher than the official results. The main reason is that we modified the `weight_decay` parameter.
4. Our experiments revealed that training for 24 epochs leads to overfitting. Therefore, we chose the best-performing model. If users want to train on a custom dataset, it is advisable to shorten the number of epochs and save the best-performing model.
5. Due to the official absence of fine-tuning hyperparameters for the GLIP-L model, we have not yet reproduced the official accuracy. I have found that overfitting can also occur, so it may be necessary to consider custom modifications to data augmentation and model enhancement. Given the high cost of training, we have not conducted any research on this matter at the moment.

## LVIS Results

|   Model    | Official | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP |       Pre-Train Data       |                                 Config                                  |                                           Download                                           |
| :--------: | :------: | :---------: | :---------: | :---------: | :--------: | :--------: | :--------: | :--------: | :-------: | :------------------------: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| GLIP-T (A) |    ✔     |             |             |             |            |            |            |            |           |            O365            | [config](lvis/glip_atss_swin-t_a_fpn_dyhead_pretrain_zeroshot_lvis.py)  | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth) |
| GLIP-T (A) |          |    12.1     |    15.5     |    25.8     |    20.2    |    6.2     |    10.9    |    22.8    |   14.7    |            O365            | [config](lvis/glip_atss_swin-t_a_fpn_dyhead_pretrain_zeroshot_lvis.py)  | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth) |
| GLIP-T (B) |    ✔     |             |             |             |            |            |            |            |           |            O365            | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth) |
| GLIP-T (B) |          |     8.6     |    13.9     |    26.0     |    19.3    |    4.6     |    9.8     |    22.6    |   13.9    |            O365            | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth) |
| GLIP-T (C) |    ✔     |    14.3     |    19.4     |    31.1     |    24.6    |            |            |            |           |         O365,GoldG         | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth) |
| GLIP-T (C) |          |    14.4     |    19.8     |    31.9     |    25.2    |    8.3     |    13.2    |    28.1    |   18.2    |         O365,GoldG         | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth) |
|   GLIP-T   |    ✔     |             |             |             |            |            |            |            |           |    O365,GoldG,CC3M,SBU     | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) |  [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth)  |
|   GLIP-T   |          |    18.1     |    21.2     |    33.1     |    26.7    |    10.8    |    14.7    |    29.0    |   19.6    |    O365,GoldG,CC3M,SBU     | [config](lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_lvis.py) |  [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth)  |
|   GLIP-L   |    ✔     |    29.2     |    34.9     |    42.1     |    37.9    |            |            |            |           | FourODs,GoldG,CC3M+12M,SBU |  [config](lvis/glip_atss_swin-l_fpn_dyhead_pretrain_zeroshot_lvis.py)   |   [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth)    |
|   GLIP-L   |          |    27.9     |    33.7     |    39.7     |    36.1    |    20.2    |    25.8    |    35.3    |   28.5    | FourODs,GoldG,CC3M+12M,SBU |  [config](lvis/glip_atss_swin-l_fpn_dyhead_pretrain_zeroshot_lvis.py)   |   [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth)    |

Note:

1. The above are zero-shot evaluation results.
2. The evaluation metric we used is LVIS FixAP. For specific details, please refer to [Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details](https://arxiv.org/pdf/2102.01066.pdf).
3. We found that the performance on small models is better than the official results, but it is lower on large models. This is mainly due to the incomplete alignment of the GLIP post-processing.

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

### Results on Flickr30k

| Model         | Official | Pre-Train Data      | Val R@1 | Val R@5 | Val R@10 | Test R@1 | Test R@5 | Test R@10 |
| ------------- | -------- | ------------------- | ------- | ------- | -------- | -------- | -------- | --------- |
| **GLIP-T(C)** | ✔        | O365, GoldG         | 84.8    | 94.9    | 96.3     | 85.5     | 95.4     | 96.6      |
| **GLIP-T(C)** |          | O365, GoldG         | 84.9    | 94.9    | 96.3     | 85.6     | 95.4     | 96.7      |
| **GLIP-T**    |          | O365,GoldG,CC3M,SBU | 85.3    | 95.5    | 96.9     | 86.0     | 95.9     | 97.2      |
