# ODinW

[ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://arxiv.org/pdf/2204.08790.pdf)

<!-- [DATASET] -->

## Get Started

1. To download dataset, you can refer to [reference document](../../docs/zh_cn/user_guides/dataset_prepare.md)

2. You can use the following data to run the inference.

```shell
cd $MMDETROOT

python tools/test.py configs/odinw35/glip_atss_swin_t_fpn_dataset.py checkpoints/glip_tiny_a_mmdet-b3654169.pth
```

## Abstract

Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets and tasks. However, it remains challenging to evaluate the transferablity of these models due to the lack of easy-to-use evaluation toolkits and public benchmarks. To tackle this, we build ELEVATER 1 , the first benchmark and toolkit for evaluating (pre-trained) language-augmented visual models. ELEVATER is composed of three components. (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to facilitate model evaluation on downstream tasks. (iii) Metrics. A variety of evaluation metrics are used to measure sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). ELEVATER is platform for Computer Vision in the Wild (CVinW), and is publicly released at https://computer-vision-in-the-wild.github.io/ELEVATER/

## Results and models of odinw13

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

Note:

1. The above are zero-shot evaluation results.
2. The config and weights of GLIPs models can be found at [here](../glip/README.md)
3. The config and weights of GroundingDINO models can be found at [here](../grounding_dino/README.md)

## Results and models of odinw35

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

Note:

1. The above are zero-shot evaluation results.
2. The config and weights of GLIPs models can be found at [here](../glip/README.md)
3. The config and weights of GroundingDINO models can be found at [here](../grounding_dino/README.md)

## Citation

```
@misc{li2022elevater,
      title={ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models},
      author={Chunyuan Li and Haotian Liu and Liunian Harold Li and Pengchuan Zhang and Jyoti Aneja and Jianwei Yang and Ping Jin and Houdong Hu and Zicheng Liu and Yong Jae Lee and Jianfeng Gao},
      year={2022},
      eprint={2204.08790},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
