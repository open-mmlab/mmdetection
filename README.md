<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

<img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

[📘Documentation](https://mmdetection.readthedocs.io/en/v2.20.0/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/v2.20.0/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/v2.20.0/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/v2.20.0/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

---

<font size=4><center><b> Table of Contents </b> </center></font>

- [Introduction](#introduction)
- [License](#license)
- [Changelog](#changelog)
- [Benchmark and model zoo](#benchmark-and-model-zoo)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Projects in OpenMMLab](#projects-in-openmmlab)

---

# Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

</details>

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

# License

This project is released under the [Apache 2.0 license](LICENSE).

# Changelog

**2.20.0** was released in 27/12/2021:

- Support [TOOD](configs/tood/README.md): Task-aligned One-stage Object Detection (ICCV 2021 Oral)
- Support resuming from the latest checkpoint automatically

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

For compatibility changes between different versions of MMDetection, please refer to [compatibility.md](docs/en/compatibility.md).

# Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<details open>
<summary>Supported backbones:</summary>

- [x] [ResNet (CVPR'2016)](mmdet/models/backbones/resnet.py)
- [x] [ResNeXt (CVPR'2017)](mmdet/models/backbones/resnext.py)
- [x] [VGG (ICLR'2015)](mmdet/models/backbones/ssd_vgg.py)
- [x] [MobileNetV2 (CVPR'2018)](mmdet/models/backbones/mobilenet_v2.py)
- [x] [HRNet (CVPR'2019)](mmdet/models/backbones/hrnet.py)
- [x] [RegNet (CVPR'2020)](mmdet/models/backbones/regnet.py)
- [x] [Res2Net (TPAMI'2020)](mmdet/models/backbones/res2net.py)
- [x] [ResNeSt (ArXiv'2020)](mmdet/models/backbones/resnest.py)
- [x] [Swin (CVPR'2021)](mmdet/models/backbones/swin.py)
- [x] [PVT (ICCV'2021)](mmdet/models/backbones/pvt.py)
- [x] [PVTv2 (ArXiv'2021)](mmdet/models/backbones/pvt.py)

</details>

<details open>
<summary>Supported methods:</summary>

- [x] [RPN (NeurIPS'2015)](configs/rpn/README.md)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn/README.md)
- [x] [Faster R-CNN (NeurIPS'2015)](configs/faster_rcnn/README.md)
- [x] [Mask R-CNN (ICCV'2017)](configs/mask_rcnn/README.md)
- [x] [Cascade R-CNN (CVPR'2018)](configs/cascade_rcnn/README.md)
- [x] [Cascade Mask R-CNN (CVPR'2018)](configs/cascade_rcnn/README.md)
- [x] [SSD (ECCV'2016)](configs/ssd/README.md)
- [x] [RetinaNet (ICCV'2017)](configs/retinanet/README.md)
- [x] [GHM (AAAI'2019)](configs/ghm/README.md)
- [x] [Mask Scoring R-CNN (CVPR'2019)](configs/ms_rcnn/README.md)
- [x] [Double-Head R-CNN (CVPR'2020)](configs/double_heads/README.md)
- [x] [Hybrid Task Cascade (CVPR'2019)](configs/htc/README.md)
- [x] [Libra R-CNN (CVPR'2019)](configs/libra_rcnn/README.md)
- [x] [Guided Anchoring (CVPR'2019)](configs/guided_anchoring/README.md)
- [x] [FCOS (ICCV'2019)](configs/fcos/README.md)
- [x] [RepPoints (ICCV'2019)](configs/reppoints/README.md)
- [x] [Foveabox (TIP'2020)](configs/foveabox/README.md)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor/README.md)
- [x] [NAS-FPN (CVPR'2019)](configs/nas_fpn/README.md)
- [x] [ATSS (CVPR'2020)](configs/atss/README.md)
- [x] [FSAF (CVPR'2019)](configs/fsaf/README.md)
- [x] [PAFPN (CVPR'2018)](configs/pafpn/README.md)
- [x] [Dynamic R-CNN (ECCV'2020)](configs/dynamic_rcnn/README.md)
- [x] [PointRend (CVPR'2020)](configs/point_rend/README.md)
- [x] [CARAFE (ICCV'2019)](configs/carafe/README.md)
- [x] [DCNv2 (CVPR'2019)](configs/dcn/README.md)
- [x] [Group Normalization (ECCV'2018)](configs/gn/README.md)
- [x] [Weight Standardization (ArXiv'2019)](configs/gn+ws/README.md)
- [x] [OHEM (CVPR'2016)](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS (ICCV'2017)](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention (ICCV'2019)](configs/empirical_attention/README.md)
- [x] [GCNet (ICCVW'2019)](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [InstaBoost (ICCV'2019)](configs/instaboost/README.md)
- [x] [GRoIE (ICPR'2020)](configs/groie/README.md)
- [x] [DetectoRS (ArXiv'2020)](configs/detectors/README.md)
- [x] [Generalized Focal Loss (NeurIPS'2020)](configs/gfl/README.md)
- [x] [CornerNet (ECCV'2018)](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization (ECCV'2020)](configs/sabl/README.md)
- [x] [YOLOv3 (ArXiv'2018)](configs/yolo/README.md)
- [x] [PAA (ECCV'2020)](configs/paa/README.md)
- [x] [YOLACT (ICCV'2019)](configs/yolact/README.md)
- [x] [CentripetalNet (CVPR'2020)](configs/centripetalnet/README.md)
- [x] [VFNet (ArXiv'2020)](configs/vfnet/README.md)
- [x] [DETR (ECCV'2020)](configs/detr/README.md)
- [x] [Deformable DETR (ICLR'2021)](configs/deformable_detr/README.md)
- [x] [CascadeRPN (NeurIPS'2019)](configs/cascade_rpn/README.md)
- [x] [SCNet (AAAI'2021)](configs/scnet/README.md)
- [x] [AutoAssign (ArXiv'2020)](configs/autoassign/README.md)
- [x] [YOLOF (CVPR'2021)](configs/yolof/README.md)
- [x] [Seasaw Loss (CVPR'2021)](configs/seesaw_loss/README.md)
- [x] [CenterNet (CVPR'2019)](configs/centernet/README.md)
- [x] [YOLOX (ArXiv'2021)](configs/yolox/README.md)
- [x] [SOLO (ECCV'2020)](configs/solo/README.md)
- [x] [QueryInst (ICCV'2021)](configs/queryinst/README.md)
- [x] [TOOD (ICCV'2021)](configs/tood/README.md)

</details>

Some other methods are also supported in [projects using MMDetection](./docs/en/projects.md).

# Installation

Please refer to [get_started.md](docs/en/get_started.md) for installation.

# Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/en/1_exist_data_model.md) and [with new dataset](docs/en/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/en/tutorials/finetune.md), [adding new dataset](docs/en/tutorials/customize_dataset.md), [designing data pipeline](docs/en/tutorials/data_pipeline.md), [customizing models](docs/en/tutorials/customize_models.md), [customizing runtime settings](docs/en/tutorials/customize_runtime.md) and [useful tools](docs/en/useful_tools.md).

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

# Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

# Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

# Citation

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

# Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A comprehensive toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
