# Data Prepare and Process

## MM-GDINO-T Pre-train Dataset

For the MM-GDINO-T model, we provide a total of 5 different data combination pre-training configurations. The data is trained in a progressive accumulation manner, so users can prepare it according to their actual needs.

### 1 Objects365v1

The corresponding training config is [grounding_dino_swin-t_pretrain_obj365](./grounding_dino_swin-t_pretrain_obj365.py)

Objects365v1 can be downloaded from [opendatalab](https://opendatalab.com/OpenDataLab/Objects365_v1). It offers two methods of download: CLI and SDK.

After downloading and unzipping, place the dataset or create a symbolic link to the `data/objects365v1` directory. The directory structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── objects365v1
│   │   ├── objects365_train.json
│   │   ├── objects365_val.json
│   │   ├── train
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── test
```

Then, use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training.

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
```

After the program runs successfully, it will create two new files, `o365v1_train_od.json` and `o365v1_label_map.json`, in the `data/objects365v1` directory. The complete structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── objects365v1
│   │   ├── objects365_train.json
│   │   ├── objects365_val.json
│   │   ├── o365v1_train_od.json
│   │   ├── o365v1_label_map.json
│   │   ├── train
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── test
```

### 2 COCO 2017

The above configuration will evaluate the performance on the COCO 2017 dataset during the training process. Therefore, it is necessary to prepare the COCO 2017 dataset. You can download it from the [COCO](https://cocodataset.org/) official website or from [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017).

After downloading and unzipping, place the dataset or create a symbolic link to the `data/coco` directory. The directory structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

### 3 GoldG

After downloading the dataset, you can start training with the [grounding_dino_swin-t_pretrain_obj365_goldg](./grounding_dino_swin-t_pretrain_obj365_goldg.py) configuration.

The GoldG dataset includes the `GQA` and `Flickr30k` datasets, which are part of the MixedGrounding dataset mentioned in the GLIP paper, excluding the COCO dataset. The download links are [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations), and the specific files currently needed are `mdetr_annotations/final_mixed_train_no_coco.json` and `mdetr_annotations/final_flickr_separateGT_train.json`.

Then download the [GQA images](https://nlp.stanford.edu/data/gqa/images.zip). After downloading and unzipping, place the dataset or create a symbolic link to them in the `data/gqa` directory, with the following directory structure:

```text
mmdetection
├── configs
├── data
│   ├── gqa
|   |   ├── final_mixed_train_no_coco.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Then download the [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/). You need to apply for access to this dataset and then download it using the provided link. After downloading and unzipping, place the dataset or create a symbolic link to them in the `data/flickr30k_entities` directory, with the following directory structure:

```text
mmdetection
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

For the GQA dataset, you need to use [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/goldg2odvg.py data/gqa/final_mixed_train_no_coco.json
```

After the program has run, a new file `final_mixed_train_no_coco_vg.json` will be created in the `data/gqa` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── gqa
|   |   ├── final_mixed_train_no_coco.json
|   |   ├── final_mixed_train_no_coco_vg.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

For the Flickr30k dataset, you need to use [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/goldg2odvg.py data/flickr30k_entities/final_flickr_separateGT_train.json
```

After the program has run, a new file `final_flickr_separateGT_train_vg.json` will be created in the `data/flickr30k_entities` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── final_flickr_separateGT_train_vg.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 4 GRIT-20M

The corresponding training configuration is [grounding_dino_swin-t_pretrain_obj365_goldg_grit9m](./grounding_dino_swin-t_pretrain_obj365_goldg_grit9m.py).

The GRIT dataset can be downloaded using the img2dataset package from [GRIT](https://huggingface.co/datasets/zzliang/GRIT#download-image). By default, the dataset size is 1.1T, and downloading and processing it may require at least 2T of disk space, depending on your available storage capacity. After downloading, the dataset is in its original format, which includes:

```text
mmdetection
├── configs
├── data
│    ├── grit_raw
│    │    ├── 00000_stats.json
│    │    ├── 00000.parquet
│    │    ├── 00000.tar
│    │    ├── 00001_stats.json
│    │    ├── 00001.parquet
│    │    ├── 00001.tar
│    │    ├── ...
```

After downloading, further format processing is required:

```shell
python tools/dataset_converters/grit_processing.py data/grit_raw data/grit_processed
```

The processed format is as follows:

```text
mmdetection
├── configs
├── data
│    ├── grit_processed
│    │    ├── annotations
│    │    │   ├── 00000.json
│    │    │   ├── 00001.json
│    │    │   ├── ...
│    │    ├── images
│    │    │   ├── 00000
│    │    │   │   ├── 000000000.jpg
│    │    │   │   ├── 000000003.jpg
│    │    │   │   ├── 000000004.jpg
│    │    │   │   ├── ...
│    │    │   ├── 00001
│    │    │   ├── ...
```

As for the GRIT dataset, you need to use [grit2odvg.py](../../tools/dataset_converters/grit2odvg.py) to convert it to the format of ODVG:

```shell
python tools/dataset_converters/grit2odvg.py data/grit_processed/
```

After the program has run, a new file `grit20m_vg.json` will be created in the `data/grit_processed` directory, which has about 9M data, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│    ├── grit_processed
|    |    ├── grit20m_vg.json
│    │    ├── annotations
│    │    │   ├── 00000.json
│    │    │   ├── 00001.json
│    │    │   ├── ...
│    │    ├── images
│    │    │   ├── 00000
│    │    │   │   ├── 000000000.jpg
│    │    │   │   ├── 000000003.jpg
│    │    │   │   ├── 000000004.jpg
│    │    │   │   ├── ...
│    │    │   ├── 00001
│    │    │   ├── ...
```

### 5 V3Det

The corresponding training configurations are:

- [grounding_dino_swin-t_pretrain_obj365_goldg_v3det](./grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py)
- [grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det](./grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py)

The V3Det dataset can be downloaded from [opendatalab](https://opendatalab.com/V3Det/V3Det). After downloading and unzipping, place the dataset or create a symbolic link to it in the `data/v3det` directory, with the following directory structure:

```text
mmdetection
├── configs
├── data
│   ├── v3det
│   │   ├── annotations
│   │   |   ├── v3det_2023_v1_train.json
│   │   ├── images
│   │   │   ├── a00000066
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Then use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/coco2odvg.py data/v3det/annotations/v3det_2023_v1_train.json -d v3det
```

After the program has run, two new files `v3det_2023_v1_train_od.json` and `v3det_2023_v1_label_map.json` will be created in the `data/v3det/annotations` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── v3det
│   │   ├── annotations
│   │   |   ├── v3det_2023_v1_train.json
│   │   |   ├── v3det_2023_v1_train_od.json
│   │   |   ├── v3det_2023_v1_label_map.json
│   │   ├── images
│   │   │   ├── a00000066
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 6 Data Splitting and Visualization

Considering that users need to prepare many datasets, which is inconvenient for confirming images and annotations before training, we provide a data splitting and visualization tool. This tool can split the dataset into a tiny version and then use a visualization script to check the correctness of the images and labels.

1. Splitting the Dataset

The script is located [here](../../tools/misc/split_odvg.py). Taking `Object365 v1` as an example, the command to split the dataset is as follows:

```shell
python tools/misc/split_odvg.py data/object365_v1/ o365v1_train_od.json train your_output_dir --label-map-file o365v1_label_map.json -n 200
```

After running the above script, it will create a folder structure in the `your_output_dir` directory identical to `data/object365_v1/`, but it will only save 200 training images and their corresponding json files for convenient user review.

2. Visualizing the Original Dataset

The script is located [here](../../tools/analysis_tools/browse_grounding_raw.py). Taking `Object365 v1` as an example, the command to visualize the dataset is as follows:

```shell
python tools/analysis_tools/browse_grounding_raw.py data/object365_v1/ o365v1_train_od.json train --label-map-file o365v1_label_map.json -o your_output_dir --not-show
```

After running the above script, it will generate images in the `your_output_dir` directory that include both the pictures and their labels, making it convenient for users to review.

3. Visualizing the Output Dataset

The script is located [here](../../tools/analysis_tools/browse_grounding_dataset.py). Users can use this script to view the results of the dataset output, including the results of data augmentation. Taking `Object365 v1` as an example, the command to visualize the dataset is as follows:

```shell
python tools/analysis_tools/browse_grounding_dataset.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py  -o your_output_dir --not-show
```

After running the above script, it will generate images in the `your_output_dir` directory that include both the pictures and their labels, making it convenient for users to review.

## MM-GDINO-L Pre-training Data Preparation and Processing

### 1 Object365 v2

Objects365_v2 can be downloaded from [opendatalab](https://opendatalab.com/OpenDataLab/Objects365). It offers two download methods: CLI and SDK.

After downloading and unzipping, place the dataset or create a symbolic link to it in the `data/objects365v2` directory, with the following directory structure:

```text
mmdetection
├── configs
├── data
│   ├── objects365v2
│   │   ├── annotations
│   │   │   ├── zhiyuan_objv2_train.json
│   │   ├── train
│   │   │   ├── patch0
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Since some category names in Objects365v2 are incorrect, it is necessary to correct them first.

```shell
python tools/dataset_converters/fix_o365_names.py
```

A new annotation file `zhiyuan_objv2_train_fixname.json` will be generated in the `data/objects365v2/annotations` directory.

Then use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v2/annotations/zhiyuan_objv2_train_fixname.json -d o365v2
```

After the program has run, two new files `zhiyuan_objv2_train_fixname_od.json` and `o365v2_label_map.json` will be created in the `data/objects365v2` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── objects365v2
│   │   ├── annotations
│   │   │   ├── zhiyuan_objv2_train.json
│   │   │   ├── zhiyuan_objv2_train_fixname.json
│   │   │   ├── zhiyuan_objv2_train_fixname_od.json
│   │   │   ├── o365v2_label_map.json
│   │   ├── train
│   │   │   ├── patch0
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 2 OpenImages v6

OpenImages v6 can be downloaded from the [official website](https://storage.googleapis.com/openimages/web/download_v6.html). Due to the large size of the dataset, it may take some time to download. After completion, the file structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── OpenImages
│   │   ├── annotations
|   │   │   ├── oidv6-train-annotations-bbox.csv
|   │   │   ├── class-descriptions-boxable.csv
│   │   ├── OpenImages
│   │   │   ├── train
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Then use [openimages2odvg.py](../../tools/dataset_converters/openimages2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/openimages2odvg.py data/OpenImages/annotations
```

After the program has run, two new files `oidv6-train-annotation_od.json` and `openimages_label_map.json` will be created in the `data/OpenImages/annotations` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── OpenImages
│   │   ├── annotations
|   │   │   ├── oidv6-train-annotations-bbox.csv
|   │   │   ├── class-descriptions-boxable.csv
|   │   │   ├── oidv6-train-annotations_od.json
|   │   │   ├── openimages_label_map.json
│   │   ├── OpenImages
│   │   │   ├── train
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 3 V3Det

Referring to the data preparation section of the previously mentioned MM-GDINO-T pre-training data preparation and processing, the complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── v3det
│   │   ├── annotations
│   │   |   ├── v3det_2023_v1_train.json
│   │   |   ├── v3det_2023_v1_train_od.json
│   │   |   ├── v3det_2023_v1_label_map.json
│   │   ├── images
│   │   │   ├── a00000066
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 4 LVIS 1.0

Please refer to the `2 LVIS 1.0` section of the later `Fine-tuning Dataset Preparation`. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_train_od.json
│   │   │   ├── lvis_v1_label_map.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   ├── lvis_od_val.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

### 5 COCO2017 OD

You can refer to the earlier section `MM-GDINO-T Pre-training Data Preparation and Processing` for data preparation. For convenience in subsequent processing, please create a symbolic link or move the downloaded [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations) folder to the `data/coco` path. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── ...
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

Due to some overlap between COCO2017 train and RefCOCO/RefCOCO+/RefCOCOg/gRefCOCO val, if not removed in advance, there will be data leakage when evaluating RefExp.

```shell
python tools/dataset_converters/remove_cocotrain2017_from_refcoco.py data/coco/mdetr_annotations data/coco/annotations/instances_train2017.json
```

A new file `instances_train2017_norefval.json` will be created in the `data/coco/annotations` directory. Finally, use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/coco2odvg.py data/coco/annotations/instances_train2017_norefval.json -d coco
```

Two new files `instances_train2017_norefval_od.json` and `coco_label_map.json` will be created in the `data/coco/annotations` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2017_norefval_od.json
│   │   │   ├── coco_label_map.json
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── ...
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

Note: There are 15,000 images that overlap between the COCO2017 train and LVIS 1.0 val datasets. Therefore, if the COCO2017 train dataset is used in training, the evaluation results of LVIS 1.0 val will have a data leakage issue. However, LVIS 1.0 minival does not have this problem.

### 6 GoldG

Please refer to the section on `MM-GDINO-T Pre-training Data Preparation and Processing`.

```text
mmdetection
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── final_flickr_separateGT_train_vg.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   ├── gqa
|   |   ├── final_mixed_train_no_coco.json
|   |   ├── final_mixed_train_no_coco_vg.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 7 COCO2014 VG

MDetr provides a Phrase Grounding version of the COCO2014 train annotations. The original annotation file is named `final_mixed_train.json`, and similar to the previous structure, the file structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── mdetr_annotations
│   │   │   ├── final_mixed_train.json
│   │   │   ├── ...
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

We can extract the COCO portion of the data from `final_mixed_train.json`.

```shell
python tools/dataset_converters/extract_coco_from_mixed.py data/coco/mdetr_annotations/final_mixed_train.json
```

A new file named `final_mixed_train_only_coco.json` will be created in the `data/coco/mdetr_annotations` directory. Finally, use [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) to convert it into the ODVG format required for training:

```shell
python tools/dataset_converters/goldg2odvg.py data/coco/mdetr_annotations/final_mixed_train_only_coco.json
```

A new file named `final_mixed_train_only_coco_vg.json` will be created in the `data/coco/mdetr_annotations` directory, with the complete structure as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── mdetr_annotations
│   │   │   ├── final_mixed_train.json
│   │   │   ├── final_mixed_train_only_coco.json
│   │   │   ├── final_mixed_train_only_coco_vg.json
│   │   │   ├── ...
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Note: COCO2014 train and COCO2017 val do not have duplicate images, so there is no need to worry about data leakage issues in COCO evaluation.

### 8 Referring Expression Comprehension

There are a total of 4 datasets included. For data preparation, please refer to the `Fine-tuning Dataset Preparation` section.

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── finetune_refcoco_testB.json
│   │   │   ├── finetune_refcoco+_testA.json
│   │   │   ├── finetune_refcoco+_testB.json
│   │   │   ├── finetune_refcocog_test.json
│   │   │   ├── finetune_refcoco_train_vg.json
│   │   │   ├── finetune_refcoco+_train_vg.json
│   │   │   ├── finetune_refcocog_train_vg.json
│   │   │   ├── finetune_grefcoco_train_vg.json
```

### 9 GRIT-20M

Please refer to the `MM-GDINO-T Pre-training Data Preparation and Processing` section.

## Preparation of Evaluation Dataset

### 1 COCO 2017

The data preparation process is consistent with the previous descriptions, and the final structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

### 2 LVIS 1.0

The LVIS 1.0 val dataset includes both mini and full versions. The significance of the mini version is:

1. The full LVIS val evaluation dataset is quite large, and conducting an evaluation with it can take a significant amount of time.
2. In the full LVIS val dataset, there are 15,000 images from the COCO2017 train dataset. If a user has used the COCO2017 data for training, there can be a data leakage issue when evaluating on the full LVIS val dataset

The LVIS 1.0 dataset contains images that are exactly the same as the COCO2017 dataset, with the addition of new annotations. You can download the minival annotation file from [here](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json), and the val 1.0 annotation file from [here](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_od_val.json). The final structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   ├── lvis_od_val.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

### 3 ODinW

ODinW, which stands for Object Detection in the Wild, is a dataset used to evaluate the generalization capability of grounding pre-trained models in different real-world scenarios. It consists of two subsets, ODinW13 and ODinW35, representing datasets composed of 13 and 35 different datasets, respectively. You can download it from [here](https://huggingface.co/GLIPModel/GLIP/tree/main/odinw_35), and then unzip each file. The final structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── odinw
│   │   ├── AerialMaritimeDrone
│   │   |   |── large
│   │   |   |   ├── test
│   │   |   |   ├── train
│   │   |   |   ├── valid
│   │   |   |── tiled
│   │   ├── AmericanSignLanguageLetters
│   │   ├── Aquarium
│   │   ├── BCCD
│   │   ├── ...
```

When evaluating ODinW35, custom prompts are required. Therefore, it's necessary to preprocess the annotated JSON files in advance. You can use the [override_category.py](./odinw/override_category.py) script for this purpose. After processing, it will generate new annotation files without overwriting the original ones.

```shell
python configs/mm_grounding_dino/odinw/override_category.py data/odinw/
```

### 4 DOD

DOD stands for Described Object Detection, and it is introduced in the paper titled [Described Object Detection: Liberating Object Detection with Flexible Expressions](https://arxiv.org/abs/2307.12813). You can download the dataset from [here](https://github.com/shikras/d-cube?tab=readme-ov-file). The final structure of the dataset is as follows:

```text
mmdetection
├── configs
├── data
│   ├── d3
│   │   ├── d3_images
│   │   ├── d3_json
│   │   ├── d3_pkl
```

### 5 Flickr30k Entities

In the previous GoldG data preparation section, we downloaded the necessary files for training with Flickr30k. For evaluation, you will need 2 JSON files, which you can download from [here](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_val.json) and [here](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_test.json). The final structure of the dataset is as follows:

```text
mmdetection
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── final_flickr_separateGT_val.json
│   │   ├── final_flickr_separateGT_test.json
│   │   ├── final_flickr_separateGT_train_vg.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### 6 Referring Expression Comprehension

Referential Expression Comprehension includes 4 datasets: RefCOCO, RefCOCO+, RefCOCOg, and gRefCOCO. The images used in these 4 datasets are from COCO2014 train, similar to COCO2017. You can download the images from the official COCO website or opendatalab. The annotations can be directly downloaded from [here](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations). The mdetr_annotations folder contains a large number of annotations, so you can choose to download only the JSON files you need. The final structure of the dataset is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── finetune_refcoco_testB.json
│   │   │   ├── finetune_refcoco+_testA.json
│   │   │   ├── finetune_refcoco+_testB.json
│   │   │   ├── finetune_refcocog_test.json
│   │   │   ├── finetune_refcocog_test.json
```

Please note that gRefCOCO is introduced in [GREC: Generalized Referring Expression Comprehension](https://arxiv.org/abs/2308.16182) and is not available in the `mdetr_annotations` folder. You will need to handle it separately. Here are the specific steps:

1. Download [gRefCOCO](https://github.com/henghuiding/gRefCOCO?tab=readme-ov-file) and unzip it into the `data/coco/` folder.

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   ├── grefs
│   │   │   ├── grefs(unc).json
│   │   │   ├── instances.json
```

2. Convert to COCO format

You can use the official [conversion script](https://github.com/henghuiding/gRefCOCO/blob/b4b1e55b4d3a41df26d6b7d843ea011d581127d4/mdetr/scripts/fine-tuning/grefexp_coco_format.py) provided by gRefCOCO. Please note that you need to uncomment line 161 and comment out line 160 in the script to obtain the full JSON file.

```shell
# you need to clone the official repo
git clone https://github.com/henghuiding/gRefCOCO.git
cd gRefCOCO/mdetr
python scripts/fine-tuning/grefexp_coco_format.py --data_path ../../data/coco/grefs --out_path ../../data/coco/mdetr_annotations/ --coco_path ../../data/coco
```

Four JSON files will be generated in the `data/coco/mdetr_annotations/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── finetune_refcoco_testB.json
│   │   │   ├── finetune_grefcoco_train.json
│   │   │   ├── finetune_grefcoco_val.json
│   │   │   ├── finetune_grefcoco_testA.json
│   │   │   ├── finetune_grefcoco_testB.json
```

## Fine-Tuning Dataset Preparation

### 1 COCO 2017

COCO is the most commonly used dataset in the field of object detection, and we aim to explore its fine-tuning modes more comprehensively. From current developments, there are a total of three fine-tuning modes:

1. Closed-set fine-tuning, where the description on the text side cannot be modified after fine-tuning, transforms into a closed-set algorithm. This approach maximizes performance on COCO but loses generality.
2. Open-set continued pretraining fine-tuning involves using pretraining methods consistent with the COCO dataset. There are two approaches to this: the first is to reduce the learning rate and fix certain modules, fine-tuning only on the COCO dataset; the second is to mix COCO data with some of the pre-trained data. The goal of both approaches is to improve performance on the COCO dataset as much as possible without compromising generalization.
3. Open-vocabulary fine-tuning involves adopting a common practice in the OVD (Open-Vocabulary Detection) domain. It divides COCO categories into base classes and novel classes. During training, fine-tuning is performed only on the base classes, while evaluation is conducted on both base and novel classes. This approach allows for the assessment of COCO OVD capabilities, with the goal of improving COCO dataset performance without compromising generalization as much as possible.

\*\*(1) Closed-set Fine-tuning \*\*

This section does not require data preparation; you can directly use the data you have prepared previously.

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

**(2) Open-set Continued Pretraining Fine-tuning**
To use this approach, you need to convert the COCO training data into ODVG format. You can use the following command for conversion:

```shell
python tools/dataset_converters/coco2odvg.py data/coco/annotations/instances_train2017.json -d coco
```

This will generate new files, `instances_train2017_od.json` and `coco2017_label_map.json`, in the `data/coco/annotations/` directory. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_train2017_od.json
│   │   │   ├── coco2017_label_map.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

Once you have obtained the data, you can choose whether to perform individual pretraining or mixed pretraining.

**(3) Open-vocabulary Fine-tuning**
For this approach, you need to convert the COCO training data into OVD (Open-Vocabulary Detection) format. You can use the following command for conversion:

```shell
python tools/dataset_converters/coco2ovd.py data/coco/
```

This will generate new files, `instances_val2017_all_2.json` and `instances_val2017_seen_2.json`, in the `data/coco/annotations/` directory. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_train2017_od.json
│   │   │   ├── instances_val2017_all_2.json
│   │   │   ├── instances_val2017_seen_2.json
│   │   │   ├── coco2017_label_map.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

You can then proceed to train and test directly using the [configuration](coco/grounding_dino_swin-t_finetune_16xb4_1x_coco_48_17.py).

### 2 LVIS 1.0

LVIS is a dataset that includes 1,203 classes, making it a valuable dataset for fine-tuning. Due to its large number of classes, it's not feasible to perform closed-set fine-tuning. Therefore, we can only use open-set continued pretraining fine-tuning and open-vocabulary fine-tuning on LVIS.

You need to prepare the LVIS training JSON files first, which you can download from [here](https://www.lvisdataset.org/dataset). We only need `lvis_v1_train.json` and `lvis_v1_val.json`. After downloading them, place them in the `data/coco/annotations/` directory, and then run the following command:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   ├── lvis_od_val.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

(1) Open-set continued pretraining fine-tuning

Convert to ODVG format using the following command:

```shell
python tools/dataset_converters/lvis2odvg.py data/coco/annotations/lvis_v1_train.json
```

It will generate new files, `lvis_v1_train_od.json` and `lvis_v1_label_map.json`, in the `data/coco/annotations/` directory, and the complete dataset structure will look like this:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_train_od.json
│   │   │   ├── lvis_v1_label_map.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   ├── lvis_od_val.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

You can directly use the provided [configuration](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis.py) for training and testing, or you can modify the configuration to mix it with some of the pretraining datasets as needed.

**(2) Open Vocabulary Fine-tuning**

Convert to OVD format using the following command:

```shell
python tools/dataset_converters/lvis2ovd.py data/coco/
```

New `lvis_v1_train_od_norare.json` and `lvis_v1_label_map_norare.json` will be generated under `data/coco/annotations/`, and the complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_train_od.json
│   │   │   ├── lvis_v1_label_map.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   ├── lvis_od_val.json
│   │   │   ├── lvis_v1_train_od_norare.json
│   │   │   ├── lvis_v1_label_map_norare.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

然Then you can directly use the [configuration](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py) for training and testing.

### 3 RTTS

RTTS is a foggy weather dataset, which contains 4,322 foggy images, including five classes: bicycle, bus, car, motorbike, and person. It can be downloaded from [here](https://drive.google.com/file/d/15Ei1cHGVqR1mXFep43BO7nkHq1IEGh1e/view), and then extracted to the `data/RTTS/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── RTTS
│   │   ├── annotations_json
│   │   ├── annotations_xml
│   │   ├── ImageSets
│   │   ├── JPEGImages
```

### 4 RUOD

RUOD is an underwater object detection dataset. You can download it from [here](https://drive.google.com/file/d/1hxtbdgfVveUm_DJk5QXkNLokSCTa_E5o/view), and then extract it to the `data/RUOD/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── RUOD
│   │   ├── Environment_pic
│   │   ├── Environmet_ANN
│   │   ├── RUOD_ANN
│   │   ├── RUOD_pic
```

### 5 Brain Tumor

Brain Tumor is a 2D detection dataset in the medical field. You can download it from [here](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp/dataset/2), please make sure to choose the `COCO JSON` format. Then extract it to the `data/brain_tumor_v2/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── brain_tumor_v2
│   │   ├── test
│   │   ├── train
│   │   ├── valid
```

### 6 Cityscapes

Cityscapes is an urban street scene dataset. You can download it from [here](https://www.cityscapes-dataset.com/) or from opendatalab, and then extract it to the `data/cityscapes/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```

After downloading, you can use the [cityscapes.py](../../tools/dataset_converters/cityscapes.py) script to generate the required JSON format.

```shell
python tools/dataset_converters/cityscapes.py data/cityscapes/
```

Three new JSON files will be generated in the annotations directory. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── cityscapes
│   │   ├── annotations
│   │   │   ├── instancesonly_filtered_gtFine_train.json
│   │   │   ├── instancesonly_filtered_gtFine_val.json
│   │   │   ├── instancesonly_filtered_gtFine_test.json
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```

### 7 People in Painting

People in Painting is an oil painting dataset that you can download from [here](https://universe.roboflow.com/roboflow-100/people-in-paintings/dataset/2). Please make sure to choose the `COCO JSON` format. After downloading, unzip the dataset to the `data/people_in_painting_v2/` folder. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── people_in_painting_v2
│   │   ├── test
│   │   ├── train
│   │   ├── valid
```

### 8 Referring Expression Comprehension

Fine-tuning for Referential Expression Comprehension is similar to what was described earlier and includes four datasets. The dataset preparation for evaluation has already been organized. The complete dataset structure is as follows:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── finetune_refcoco_testB.json
│   │   │   ├── finetune_refcoco+_testA.json
│   │   │   ├── finetune_refcoco+_testB.json
│   │   │   ├── finetune_refcocog_test.json
│   │   │   ├── finetune_refcocog_test.json
```

Then we need to convert it to the required ODVG format. Please use the [refcoco2odvg.py](../../tools/dataset_converters/refcoco2odvg.py) script to perform the conversion.

```shell
python tools/dataset_converters/refcoco2odvg.py data/coco/mdetr_annotations
```

The converted dataset structure will include 4 new JSON files in the `data/coco/mdetr_annotations` directory. Here is the structure of the converted dataset:

```text
mmdetection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2014.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── train2014
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── mdetr_annotations
│   │   │   ├── final_refexp_val.json
│   │   │   ├── finetune_refcoco_testA.json
│   │   │   ├── finetune_refcoco_testB.json
│   │   │   ├── finetune_refcoco+_testA.json
│   │   │   ├── finetune_refcoco+_testB.json
│   │   │   ├── finetune_refcocog_test.json
│   │   │   ├── finetune_refcoco_train_vg.json
│   │   │   ├── finetune_refcoco+_train_vg.json
│   │   │   ├── finetune_refcocog_train_vg.json
│   │   │   ├── finetune_grefcoco_train_vg.json
```
