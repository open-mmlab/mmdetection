# Detecting Twenty-thousand Classes using Image-level Supervision

## Description

**Detic**: A **Det**ector with **i**mage **c**lasses that can use image-level labels to easily train detectors.

<p align="center"> <img src='https://github.com/facebookresearch/Detic/blob/main/docs/teaser.jpeg?raw=true' align="center" height="300px"> </p>

> [**Detecting Twenty-thousand Classes using Image-level Supervision**](http://arxiv.org/abs/2201.02605),
> Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, Ishan Misra,
> *ECCV 2022 ([arXiv 2201.02605](http://arxiv.org/abs/2201.02605))*

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

## Installation

Detic requires to install CLIP.

```shell
pip install git+https://github.com/openai/CLIP.git
```

### Demo

#### Inference with existing dataset vocabulary embeddings

First, go to the Detic project folder.

```shell
cd projects/Detic
```

Then, download the pre-computed CLIP embeddings from [dataset metainfo](https://github.com/facebookresearch/Detic/tree/main/datasets/metadata) to the `datasets/metadata` folder.
The CLIP embeddings will be loaded to the zero-shot classifier during inference.
For example, you can download LVIS's class name embeddings with the following command:

```shell
wget -P datasets/metadata https://raw.githubusercontent.com/facebookresearch/Detic/main/datasets/metadata/lvis_v1_clip_a%2Bcname.npy
```

You can run demo like this:

```shell
python demo.py \
  ${IMAGE_PATH} \
  ${CONFIG_PATH} \
  ${MODEL_PATH} \
  --show \
  --score-thr 0.5 \
  --dataset lvis
```

![image](https://user-images.githubusercontent.com/12907710/213624759-f0a2ba0c-0f5c-4424-a350-5ba5349e5842.png)

### Inference with custom vocabularies

- Detic can detects any class given class names by using CLIP.

You can detect custom classes with `--class-name` command:

```
python demo.py \
  ${IMAGE_PATH} \
  ${CONFIG_PATH} \
  ${MODEL_PATH} \
  --show \
  --score-thr 0.3 \
  --class-name headphone webcam paper coffe
```

![image](https://user-images.githubusercontent.com/12907710/213624637-e9e8a313-9821-4782-a18a-4408c876852b.png)

Note that `headphone`, `paper` and `coffe` (typo intended) are not LVIS classes. Despite the misspelled class name, Detic can produce a reasonable detection for `coffe`.

## Results

Here we only provide the Detic Swin-B model for the open vocabulary demo. Multi-dataset training and open-vocabulary testing will be supported in the future.

To find more variants, please visit the [official model zoo](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md).

| Backbone |       Training data        |                                Config                                 |                                                                                      Download                                                                                      |
| :------: | :------------------------: | :-------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-B  | ImageNet-21K & LVIS & COCO | [config](./configs/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth) |

## Citation

If you find Detic is useful in your research or applications, please consider giving a star 🌟 to the [official repository](https://github.com/facebookresearch/Detic) and citing Detic by the following BibTeX entry.

```BibTeX
@inproceedings{zhou2022detecting,
  title={Detecting Twenty-thousand Classes using Image-level Supervision},
  author={Zhou, Xingyi and Girdhar, Rohit and Joulin, Armand and Kr{\"a}henb{\"u}hl, Philipp and Misra, Ishan},
  booktitle={ECCV},
  year={2022}
}

```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/mmdet/datasets/transforms/transforms.py#L41-L169) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/tests/test_datasets/test_transforms/test_transforms.py#L35-L88) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
