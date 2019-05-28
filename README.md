
# mmdetection

## Introduction

The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)

### Major features

- **Modular Design**

  One can easily construct a customized object detection framework by combining different components.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **Efficient**

  All basic bbox and mask operations run on GPUs now.
  The training speed is nearly 2x faster than Detectron and comparable to maskrcnn-benchmark.

- **State of the art**

  This was the codebase of the *MMDet* team, who won the [COCO Detection 2018 challenge](http://cocodataset.org/#detection-leaderboard).

Apart from mmdetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research,
which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Updates

v0.6.0 (14/04/2019)
- Up to 30% speedup compared to the model zoo.
- Support both PyTorch stable and nightly version.
- Replace NMS and SigmoidFocalLoss with Pytorch CUDA extensions.

v0.6rc0(06/02/2019)
- Migrate to PyTorch 1.0.

v0.5.7 (06/02/2019)
- Add support for Deformable ConvNet v2. (Many thanks to the authors and [@chengdazhi](https://github.com/chengdazhi))
- This is the last release based on PyTorch 0.4.1.

v0.5.6 (17/01/2019)
- Add support for Group Normalization.
- Unify RPNHead and single stage heads (RetinaHead, SSDHead) with AnchorHead.

v0.5.5 (22/12/2018)
- Add SSD for COCO and PASCAL VOC.
- Add ResNeXt backbones and detection models.
- Refactoring for Samplers/Assigners and add OHEM.
- Add VOC dataset and evaluation scripts.

v0.5.4 (27/11/2018)
- Add SingleStageDetector and RetinaNet.

v0.5.3 (26/11/2018)
- Add Cascade R-CNN and Cascade Mask R-CNN.
- Add support for Soft-NMS in config files.

v0.5.2 (21/10/2018)
- Add support for custom datasets.
- Add a script to convert PASCAL VOC annotations to the expected format.

v0.5.1 (20/10/2018)
- Add BBoxAssigner and BBoxSampler, the `train_cfg` field in config files are restructured.
- `ConvFCRoIHead` / `SharedFCRoIHead` are renamed to `ConvFCBBoxHead` / `SharedFCBBoxHead` for consistency.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      |
|--------------------|:--------:|:--------:|:--------:|:--------:|
| RPN                | ✓        | ✓        | ☐        | ✗        |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        |
| SSD                | ✗        | ✗        | ✗        | ✓        |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        |
| Hybrid Task Cascade| ✓        | ✓        | ☐        | ✗        |
| FCOS               | ✓        | ✓        | ☐        | ✗        |
| Libra R-CNN        | ✓        | ✓        | ☐        | ✗        |

Other features
- [x] DCNv2
- [x] Group Normalization
- [x] Weight Standardization
- [x] OHEM
- [x] Soft-NMS
- [ ] Mixed Precision (FP16) Training (coming soon)


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve mmdetection. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.


## Citation

If you use our codebase or models in your research, please cite this project.
We will release a paper or technical report later.

```
@misc{mmdetection2018,
  author =       {Kai Chen and Jiangmiao Pang and Jiaqi Wang and Yu Xiong and Xiaoxiao Li
                  and Shuyang Sun and Wansen Feng and Ziwei Liu and Jianping Shi and
                  Wanli Ouyang and Chen Change Loy and Dahua Lin},
  title =        {mmdetection},
  howpublished = {\url{https://github.com/open-mmlab/mmdetection}},
  year =         {2018}
}
```
