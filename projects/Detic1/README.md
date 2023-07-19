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

## Demo

Here we provide the Detic model for the open vocabulary demo.  This model is trained on combined LVIS-COCO and ImageNet-21K for better demo purposes. LVIS models do not detect persons well due to its federated annotation protocol. LVIS+COCO models give better visual results.

| Backbone |         Training data          |                                Config                                 |                                                                                      Download                                                                                      |
| :------: | :----------------------------: | :-------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-B  | LVIS  &  COCO  &  ImageNet-21K | [config](./configs/detic_centernet2_swin-b_fpn_4x_lvis_coco_in21k.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth) |

You can also download other models from  [official model zoo](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md), and convert the format by run

```shell
python tools/model_converters/detic_to_mmdet.py --src /path/to/detic_weight.pth --dst /path/to/mmdet_weight.pth
```

### Inference with existing dataset vocabulary

You can detect classes of existing dataset  with `--texts` command:

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

### Inference with custom vocabularies

Detic can detects any class given class names by using CLIP. You can detect customized classes with `--texts` command:

```shell
python demo/image_demo.py \
  ${IMAGE_PATH} \
  ${CONFIG_PATH} \
  ${MODEL_PATH} \
  --texts 'headphone . webcam . paper . coffe.' \
  --pred-score-thr 0.3 \
  --palette 'random'
```

![image](https://user-images.githubusercontent.com/12907710/213624637-e9e8a313-9821-4782-a18a-4408c876852b.png)

Note that `headphone`, `paper` and `coffe` (typo intended) are not LVIS classes. Despite the misspelled class name, Detic can produce a reasonable detection for `coffe`.

## Results

### Testing

To evaluate a model with a trained model, run

```shell
python tools/test.py path/to/config.py /path/to/weight.pth
```

### Open-vocabulary LVIS

| Backbone |          Training data          | mask mAP | mask mAP_novel |                                   Config                                   | Download |
| :------: | :-----------------------------: | :------: | :------------: | :------------------------------------------------------------------------: | :------: |
| ResNet50 | LVIS-Base  &  ImageNet-21K-LVIS |   32.4   |      25.2      |  [config](./configs/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis.py)   |          |
|  Swin-B  | LVIS-Base  &  ImageNet-21K-LVIS |   40.7   |      34.0      | [config](./configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py) |          |

### Standard LVIS

| Backbone |      Training data       | mask mAP | mask mAP_novel |                                Config                                 | Download |
| :------: | :----------------------: | :------: | :------------: | :-------------------------------------------------------------------: | :------: |
| ResNet50 | LVIS & ImageNet-21K-LVIS |   33.2   |      29.7      |  [config](./configs/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py)   |          |
|  Swin-B  | LVIS & ImageNet-21K-LVIS |   41.7   |      41.7      | [config](./configs/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis.py) |          |

#### Note:

- The open-vocabulary LVIS setup is LVIS without rare class annotations in training, termed `LVIS-Base`. We evaluate rare classes as novel classes in testing.
- ` ImageNet-21K-LVIS` denotes that the model use the overlap classes between ImageNet-21K and LVIS as image-labeled data.

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
