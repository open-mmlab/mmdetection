# MMDetection

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## Introduction

The master branch works with **PyTorch 1.1 to 1.4**.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs now. The training speed is faster than or comparable to other codebases, including [Detectron](https://github.com/facebookresearch/Detectron), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v1.1.0 was released in 24/2/2020.
Please refer to [CHANGELOG.md](docs/CHANGELOG.md) for details and release history.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](docs/MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      | HRNet |
|--------------------|:--------:|:--------:|:--------:|:--------:|:-----:|
| RPN                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| SSD                | ✗        | ✗        | ✗        | ✓        | ✗     |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        | ✓     |
| GHM                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask Scoring R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| Double-Head R-CNN  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Grid R-CNN (Plus)  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Hybrid Task Cascade| ✓        | ✓        | ☐        | ✗        | ✓     |
| Libra R-CNN        | ✓        | ✓        | ☐        | ✗        | ✓     |
| Guided Anchoring   | ✓        | ✓        | ☐        | ✗        | ✓     |
| FCOS               | ✓        | ✓        | ☐        | ✗        | ✓     |
| RepPoints          | ✓        | ✓        | ☐        | ✗        | ✓     |
| Foveabox           | ✓        | ✓        | ☐        | ✗        | ✓     |
| FreeAnchor         | ✓        | ✓        | ☐        | ✗        | ✓     |
| NAS-FPN            | ✓        | ✓        | ☐        | ✗        | ✓     |
| ATSS               | ✓        | ✓        | ☐        | ✗        | ✓     |

Other features
- [x] [CARAFE](configs/carafe/README.md)
- [x] [DCNv2](configs/dcn/README.md)
- [x] [Group Normalization](configs/gn/README.md)
- [x] [Weight Standardization](configs/gn+ws/README.md)
- [x] [OHEM](configs/faster_rcnn_ohem_r50_fpn_1x.py)
- [x] Soft-NMS
- [x] [Generalized Attention](configs/empirical_attention/README.md)
- [x] [GCNet](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training](configs/fp16)
- [x] [InstaBoost](configs/instaboost/README.md)


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```


## Contact

This repo is currently maintained by Kai Chen ([@hellock](http://github.com/hellock)), Yuhang Cao ([@yhcao6](https://github.com/yhcao6)), Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)),
Jiarui Xu ([@xvjiarui](https://github.com/xvjiarui)). Other core developers include Jiangmiao Pang ([@OceanPang](https://github.com/OceanPang)) and Jiaqi Wang ([@myownskyW7](https://github.com/myownskyW7)).
