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

## Prepare Datasets

It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to `$MMDETECTION/data` as below. If your folder structure is different, you may need to change the corresponding paths in config files.

### LVIS

LVIS dataset is adopted as box-labeled data,  [LVIS](https://www.lvisdataset.org/) is available from official website or mirror.  You need to generate `lvis_v1_train_norare.json` according to the [official prepare datasets](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#coco-and-lvis) for open-vocabulary LVIS, which removes the labels of 337 rare-class from training. You can also download [lvis_v1_train_norare.json](https://download.openmmlab.com/mmdetection/v3.0/detic/data/lvis/annotations/lvis_v1_train_norare.json) from our backup. The directory should be like this.

```shell
mmdetection
├── data
│   ├── lvis
│   │   ├── annotations
│   │   |	├── lvis_v1_train.json
│   │   |	├── lvis_v1_val.json
│   │   |	├── lvis_v1_train_norare.json
│   │   ├── train2017
│   │   ├── val2017
```

### ImageNet-LVIS

ImageNet-LVIS is adopted as image-labeled data. You can download [ImageNet-21K](https://www.image-net.org/download.php) dataset from the official website.  Then you need to unzip the overlapping classes of LVIS and convert them into LVIS annotation format according to the [official prepare datasets](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#imagenet-21k). The directory should be like this.

```shell
mmdetection
├── data
│   ├── imagenet
│   │   ├── annotations
│   │   |	├── imagenet_lvis_image_info.json
│   │   ├── ImageNet-21K
│   │   |	├── n00007846
│   │   |	├── n01318894
│   │   |	├── ...
```

### Metadata

`data/metadata/` is the preprocessed meta-data (included in the repo). Please follow the [official instruction](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#metadata) to pre-process the  LVIS dataset. You will generate `lvis_v1_train_cat_info.json` for Federated loss, which contains the frequency of each category of training set of LVIS. In addition, `lvis_v1_clip_a+cname.npy` is the pre-computed CLIP embeddings for each category of LVIS. You can also choose to directly download [lvis_v1_train_cat_info](https://download.openmmlab.com/mmdetection/v3.0/detic/data/metadata/lvis_v1_train_cat_info.json) and [lvis_v1_clip_a+cname.npy](https://download.openmmlab.com/mmdetection/v3.0/detic/data/metadata/lvis_v1_clip_a%2Bcname.npy) form our backup. The directory should be like this.

```shell
mmdetection
├── data
│   ├── metadata
│   │   ├── lvis_v1_train_cat_info.json
│   │   ├── lvis_v1_clip_a+cname.npy
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

## Models and Results

### Training

There are two stages in the whole training process. The first stage is to train a model using images with box labels as the baseline. The second stage is to finetune from the baseline model and leverage image-labeled data.

#### First stage

To train the baseline with box-supervised, run

```shell
bash ./tools/dist_train.sh projects/Detic_new/detic_centernet2_r50_fpn_4x_lvis_boxsup.py 8
```

|                                         Model (Config)                                          | mask mAP | mask mAP(official) | mask mAP_rare | mask mAP_rare(officical) |
| :---------------------------------------------------------------------------------------------: | :------: | :----------------: | :-----------: | :----------------------: |
| [detic_centernet2_r50_fpn_4x_lvis_boxsup](./configs/detic_centernet2_r50_fpn_4x_lvis_boxsup.py) |   31.6   |        31.5        |     26.6      |           25.6           |

#### Second stage

The second stage uses  both object detection and image classification datasets.

##### Multi-Datasets Config

We provide improved dataset_wrapper `ConcatDataset` to concatenate multiple datasets, all datasets could have different annotation types and different pipelines (e.g., image_size). You can also obtain the index of `dataset_source` for each sample through ` get_dataset_source` . We provide sampler `MultiDataSampler` to custom the ratios of different datasets. Beside, we provide batch_sampler `MultiDataAspectRatioBatchSampler` to enable different datasets to have different batchsizes.  The config of multiple datasets is as follows:

```python
dataset_det = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='LVISV1Dataset',
        data_root='data/lvis/',
        ann_file='annotations/lvis_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_det,
        backend_args=backend_args))

dataset_cls = dict(
    type='ImageNetLVISV1Dataset',
    data_root='data/imagenet',
    ann_file='annotations/imagenet_lvis_image_info.json',
    data_prefix=dict(img='ImageNet-LVIS/'),
    pipeline=train_pipeline_cls,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=[8, 32],
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='MultiDataSampler',
        dataset_ratio=[1, 4]),
    batch_sampler=dict(
        type='MultiDataAspectRatioBatchSampler',
        num_datasets=2),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_det, dataset_cls]))
```

###### Note:

- If the one of the  multiple datasets is `ConcatDataset` , it is still considered as a dataset for `num_datasets` in `MultiDataAspectRatioBatchSampler`.

To finetune the baseline model with image-labeled data， run:

```shell
bash ./tools/dist_train.sh projects/Detic_new/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py 8
```

|                                             Model (Config)                                              | mask mAP | mask mAP(official) | mask mAP_rare | mask mAP_rare(officical) |
| :-----------------------------------------------------------------------------------------------------: | :------: | :----------------: | :-----------: | :----------------------: |
| [detic_centernet2_r50_fpn_4x_lvis_in21k-lvis](./configs/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py) |   32.9   |        33.2        |     30.9      |           29.7           |

#### Standard LVIS Results

|                                                Model (Config)                                                 | mask mAP | mask mAP(official) | mask mAP_rare | mask mAP_rare(officical) |                                                                                                                                                                                     Download                                                                                                                                                                                     |
| :-----------------------------------------------------------------------------------------------------------: | :------: | :----------------: | :-----------: | :----------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        [detic_centernet2_r50_fpn_4x_lvis_boxsup](./configs/detic_centernet2_r50_fpn_4x_lvis_boxsup.py)        |   31.6   |        31.5        |     26.6      |           25.6           |               [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis_boxsup/detic_centernet2_r50_fpn_4x_lvis_boxsup_20230911_233514-54116677.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis_boxsup/detic_centernet2_r50_fpn_4x_lvis_boxsup_20230911_233514.log.json)               |
|    [detic_centernet2_r50_fpn_4x_lvis_in21k-lvis](./configs/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py)    |   32.9   |        33.2        |     30.9      |           29.7           |       [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis_20230912_040619-9e7a3258.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis_20230912_040619.log.json)       |
|     [detic_centernet2_swin-b_fpn_4x_lvis_boxsup](./configs/detic_centernet2_swin-b_fpn_4x_lvis_boxsup.py)     |   40.7   |        40.7        |     38.0      |           35.9           |         [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis_boxsup/detic_centernet2_swin-b_fpn_4x_lvis_boxsup_20230825_061737-328e85f9.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis_boxsup/detic_centernet2_swin-b_fpn_4x_lvis_boxsup_20230825_061737.log.json)         |
| [detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis](./configs/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis.py) |   41.7   |        41.7        |     41.7      |           41.7           | [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis_20230926_235410-0c152391.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis_20230926_235410.log.json) |

#### Open-vocabulary LVIS Results

|                                                  Model (Config)                                                   | mask mAP | mask mAP(official) | mask mAP_rare | mask mAP_rare(officical) |                                                                                                                                                                                         Download                                                                                                                                                                                         |
| :---------------------------------------------------------------------------------------------------------------: | :------: | :----------------: | :-----------: | :----------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     [detic_centernet2_r50_fpn_4x_lvis-base_boxsup](./configs/detic_centernet2_r50_fpn_4x_lvis-base_boxsup.py)     |   30.4   |        30.2        |     16.2      |           16.4           |         [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis-base_boxsup/detic_centernet2_r50_fpn_4x_lvis-base_boxsup_20230921_180638-c1685ee2.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis-base_boxsup/detic_centernet2_r50_fpn_4x_lvis-base_boxsup_20230921_180638.log.json)         |
| [detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis](./configs/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis.py) |   32.6   |        32.4        |     27.4      |           24.9           | [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis_20230925_014315-2d2cc8b7.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis/detic_centernet2_r50_fpn_4x_lvis-base_in21k-lvis_20230925_014315.log.json) |

### Testing

#### Test Command

To evaluate a model with a trained model, run

```shell
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

#### Open-vocabulary LVIS Results

The models are converted from the official model zoo.

|                                                     Model (Config)                                                      | mask mAP | mask mAP_novel |                                                                                      Download                                                                                       |
| :---------------------------------------------------------------------------------------------------------------------: | :------: | :------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     [detic_centernet2_swin-b_fpn_4x_lvis-base_boxsup](./configs/detic_centernet2_swin-b_fpn_4x_lvis-base_boxsup.py)     |   38.4   |      21.9      |     [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-base_boxsup/detic_centernet2_swin-b_fpn_4x_lvis-base_boxsup-481281c8.pth)     |
| [detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis](./configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py) |   40.7   |      34.0      | [model](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-ec91245d.pth) |

###### Note:

- The open-vocabulary LVIS setup is LVIS without rare class annotations in training, termed `lvisbase`. We evaluate rare classes as novel classes in testing.
- ` in21k-lvis` denotes that the model use the overlap classes between ImageNet-21K and LVIS as image-labeled data.

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
