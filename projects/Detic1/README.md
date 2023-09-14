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

## Models and Results

### Training

#### Prepare Datasets

We use LVIS as bboxes-labeled data, and adopt the overlap classes between ImageNet-21K and LVIS, term `in21k-lvis`, as image-labeled data.  We prepare `data/imagenet/ImageNet-LVIS`  and create annotations `data/imagenet/annotations/imagenet_lvis_image_info.json` following \[official prepare_datsets\]([Detic/datasets/README.md at main · facebookresearch/Detic (github.com)](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md)). The `data` floder should look like:

```
data/
	lvis/
		train2017
		val2017
		anntations/
			lvis_v1_train.json
			lvis_v1_val.json
    imagenet/
        ImageNet-21K/
            n00007846
            n01318894
            ...
        annotations/
            imagenet_lvis_image_info.json
```

#### Multi-Datasets Config

We provide dataset_wrapper `MultiDataDataset` to concatenate multiple datasets, all datasets could have different annotation types and different pipelines (e.g., image_size). You can also obtain the index of `dataset_source` for each sample through ` get_data_info` in  `MultiDataDataset` . We provide sampler `MultiDataSampler` to custom the ratios of different datasets. Beside, we provide batch_sampler `MDAspectRatioBatchSampler` to enable different datasets to have different batchsizes.  The config of multiple datasets is as follows:

```python
dataset_det=dict(
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

dataset_cls=dict(
        type='IMAGENETLVISV1Dataset',
        data_root='data/imagenet',
        ann_file='annotations/imagenet_lvis_image_info.json',
        data_prefix=dict(img='ImageNet-LVIS/'),
        pipeline=train_pipeline_cls,
        backend_args=backend_args)

# custom ratios and batchsizes of different datasets
train_dataloader = dict(
    batch_size=[8, 32],
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='MultiDataSampler',
                 dataset_ratio=[1, 4]),
    batch_sampler=dict(type='MDAspectRatioBatchSampler',
                       num_datasets=2),
    dataset=dict(
        type='MultiDataDataset',
        datasets=[dataset_det, dataset_cls])
)
```

#### Train Command

To train a model, run

```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

#### Standard LVIS Results

|                                             Model (Config)                                              | mask mAP | mask mAP(official) | mask mAP_rare | mask mAP_rare(officical) | Download |
| :-----------------------------------------------------------------------------------------------------: | :------: | :----------------: | :-----------: | :----------------------: | :------: |
|           [boxsup_centernet2_r50_fpn_4x_lvis](./configs/boxsup_centernet2_r50_fpn_4x_lvis.py)           |   31.6   |        31.5        |     26.7      |           25.6           |          |
| [detic_centernet2_r50_fpn_4x_lvis_in21k-lvis](./configs/detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py) |   32.9   |        33.2        |     30.9      |           29.7           |          |

### Testing

#### Test Command

To evaluate a model with a trained model, run

```shell
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

#### Open-vocabulary LVIS Results

|                                                    Model (Config)                                                     | mask mAP | mask mAP_novel |
| :-------------------------------------------------------------------------------------------------------------------: | :------: | :------------: |
|    [detic_centernet2_r50_fpn_4x_lvisbase_in21k-lvis](./configs/detic_centernet2_r50_fpn_4x_lvisbase_in21k-lvis.py)    |   32.4   |      25.2      |
| [detic_centernet2_swin-b_fpn_4x_lvisbase_in21k-lvis](./configs/detic_centernet2_swin-b_fpn_4x_lvisbase_in21k-lvis.py) |   40.7   |      34.0      |

#### Note:

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
