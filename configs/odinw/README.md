# ODinW

[ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://arxiv.org/pdf/2204.08790.pdf)

## Abstract

Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets and tasks. However, it remains challenging to evaluate the transferablity of these models due to the lack of easy-to-use evaluation toolkits and public benchmarks. To tackle this, we build ELEVATER 1 , the first benchmark and toolkit for evaluating (pre-trained) language-augmented visual models. ELEVATER is composed of three components. (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to facilitate model evaluation on downstream tasks. (iii) Metrics. A variety of evaluation metrics are used to measure sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). ELEVATER is platform for Computer Vision in the Wild (CVinW), and is publicly released at https://computer-vision-in-the-wild.github.io/ELEVATER/

## Results and models of odinw13

| Method                | GLIP-a                                                       | GLIP-b                                                       | GLIP-c                                                       |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Train Set             | o364                                                         | o365                                                         | o365, goldg                                                  |
| AerialMaritimeDrone   | 0.123                                                        | 0.110                                                        | 0.130                                                        |
| Aquarium              | 0.175                                                        | 0.173                                                        | 0.191                                                        |
| CottontailRabbits     | 0.686                                                        | 0.688                                                        | 0.744                                                        |
| EgoHands              | 0.013                                                        | 0.003                                                        | 0.314                                                        |
| NorthAmericaMushrooms | 0.502                                                        | 0.367                                                        | 0.297                                                        |
| Packages              | 0.589                                                        | 0.083                                                        | 0.699                                                        |
| PascalVOC             | 0.512                                                        | 0.541                                                        | 0.565                                                        |
| pistols               | 0.339                                                        | 0.502                                                        | 0.503                                                        |
| pothole               | 0.007                                                        | 0.030                                                        | 0.058                                                        |
| Raccoon               | 0.075                                                        | 0.285                                                        | 0.241                                                        |
| ShellfishOpenImages   | 0.372                                                        | 0.337                                                        | 0.300                                                        |
| thermalDogsAndPeople  | 0.372                                                        | 0.475                                                        | 0.510                                                        |
| VehiclesOpenImages    | 0.574                                                        | 0.562                                                        | 0.549                                                        |
| average               | **0.334**                                                    | **0.320**                                                    | **0.392**                                                    |
| Config                | [config](configs/odinw/glip_atss_swin-t_a_fpn_dyhead_pretrain_odinw13.py) | [config](configs/odinw/glip_atss_swin-t_bc_fpn_dyhead_pretrain_odinw13.py) | [config](configs/odinw/glip_atss_swin-t_bc_fpn_dyhead_pretrain_odinw35.py) |
| Download              | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth) |



## Results and models of odinw35

| Method                      | GLIP-a                                                       | GLIP-b                                                       | GLIP-c                                                       |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Train Set                   | o364                                                         | o365                                                         | o365, goldg                                                  |
| AerialMaritimeDrone_large   | 0.123                                                        | 0.110                                                        | 0.130                                                        |
| AerialMaritimeDrone_tiled   | 0.174                                                        | 0.172                                                        | 0.172                                                        |
| AmericanSignLanguageLetters | 0.001                                                        | 0.003                                                        | 0.009                                                        |
| Aquarium                    | 0.175                                                        | 0.173                                                        | 0.192                                                        |
| BCCD                        | 0.016                                                        | 0.001                                                        | 0.000                                                        |
| boggleBoards                | 0.000                                                        | 0.000                                                        | 0.000                                                        |
| brackishUnderwater          | 0.016                                                        | 0.021                                                        | 0.020                                                        |
| ChessPieces                 | 0.001                                                        | 0.000                                                        | 0.001                                                        |
| CottontailRabbits           | 0.710                                                        | 0.683                                                        | 0.752                                                        |
| dice                        | 0.005                                                        | 0.004                                                        | 0.004                                                        |
| DroneControl                | 0.016                                                        | 0.006                                                        | 0.005                                                        |
| EgoHands_generic            | 0.009                                                        | 0.005                                                        | 0.510                                                        |
| EgoHands_specific           | 0.001                                                        | 0.004                                                        | 0.003                                                        |
| HardHatWorkers              | 0.029                                                        | 0.023                                                        | 0.033                                                        |
| MaskWearing                 | 0.007                                                        | 0.003                                                        | 0.005                                                        |
| MountainDewCommercial       | 0.218                                                        | 0.199                                                        | 0.478                                                        |
| NorthAmericaMushrooms       | 0.502                                                        | 0.450                                                        | 0.497                                                        |
| openPoetryVision            | 0.000                                                        | 0.000                                                        | 0.000                                                        |
| OxfordPets_by_breed         | 0.001                                                        | 0.002                                                        | 0.001                                                        |
| OxfordPets_by_species       | 0.016                                                        | 0.012                                                        | 0.013                                                        |
| PKLot                       | 0.002                                                        | 0.000                                                        | 0.000                                                        |
| Packages                    | 0.569                                                        | 0.279                                                        | 0.712                                                        |
| PascalVOC                   | 0.512                                                        | 0.541                                                        | 0.565                                                        |
| pistols                     | 0.339                                                        | 0.502                                                        | 0.503                                                        |
| plantdoc                    | 0.002                                                        | 0.007                                                        | 0.009                                                        |
| pothole                     | 0.007                                                        | 0.024                                                        | 0.085                                                        |
| Raccoons                    | 0.075                                                        | 0.285                                                        | 0.241                                                        |
| selfdrivingCar              | 0.071                                                        | 0.074                                                        | 0.081                                                        |
| ShellfishOpenImages         | 0.253                                                        | 0.337                                                        | 0.300                                                        |
| ThermalCheetah              | 0.028                                                        | 0.000                                                        | 0.028                                                        |
| thermalDogsAndPeople        | 0.372                                                        | 0.475                                                        | 0.510                                                        |
| UnoCards                    | 0.000                                                        | 0.000                                                        | 0.002                                                        |
| VehiclesOpenImages          | 0.574                                                        | 0.562                                                        | 0.549                                                        |
| WildfireSmoke               | 0.000                                                        | 0.000                                                        | 0.017                                                        |
| websiteScreenshots          | 0.003                                                        | 0.003                                                        | 0.005                                                        |
| Average                     | **0.134**                                                    | **0.138**                                                    | **0.179**                                                    |
| Config                      | [config](configs/odinw/glip_atss_swin-t_a_fpn_dyhead_pretrain_odinw35.py) | [config](configs/odinw/glip_atss_swin-t_bc_fpn_dyhead_pretrain_odinw35.py) | [config](configs/odinw/glip_atss_swin-t_bc_fpn_dyhead_pretrain_odinw35.py) |
| Download                    | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth) | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth) |

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

