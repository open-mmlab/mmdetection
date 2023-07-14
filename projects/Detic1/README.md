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

#### Inference with existing dataset vocabulary

You can run demo like this:

```shell
python demo/image_demo.py \
  ${IMAGE_PATH} \
  ${CONFIG_PATH} \
  ${MODEL_PATH} \
  --texts lvis \
  --pred-score-thr 0.5 \
  --palette 'random'
```

![image](https://user-images.githubusercontent.com/12907710/213624759-f0a2ba0c-0f5c-4424-a350-5ba5349e5842.png)

#### Inference with custom vocabularies

- Detic can detects any class given class names by using CLIP.

You can detect custom classes with `--texts` and `--custom-entities`command:

```
python demo/image_demo.py \
  ${IMAGE_PATH} \
  ${CONFIG_PATH} \
  ${MODEL_PATH} \
  --texts 'headphone . webcam . paper . coffe.' \
  --custom-entities \
  --pred-score-thr 0.5 \
  --palette 'random'
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
