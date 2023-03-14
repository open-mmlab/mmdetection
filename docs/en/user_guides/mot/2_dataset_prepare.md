## Dataset Preparation

This page provides the instructions for dataset preparation on existing benchmarks, include

- Multiple Object Tracking
  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)
  - [LVIS](https://www.lvisdataset.org/)
  - [TAO](https://taodataset.org/)
  - [DanceTrack](https://dancetrack.github.io)
- Video Instance Segmentation
  - [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### 1. Download Datasets

Please download the datasets from the official websites. It is recommended to symlink the root of the datasets to `$MMDETECTION/data`.

#### 1.1 Multiple Object Tracking

- For the training and testing of multi object tracking task, one of the MOT Challenge datasets (e.g. MOT17, TAO and DanceTrack) are needed, CrowdHuman and LVIS can be served as comlementary dataset.

- The `annotations` under `tao` contains the official annotations from [here](https://github.com/TAO-Dataset/annotations).

- The `annotations` under `lvis` contains the official annotations of lvis-v0.5 which can be downloaded according to [here](https://github.com/lvis-dataset/lvis-api/issues/23#issuecomment-894963957). The synset mapping file `coco_to_lvis_synset.json` used in `./tools/dataset_converters/tao/merge_coco_with_lvis.py` script can be found [here](https://github.com/TAO-Dataset/tao/tree/master/data).

- For users in China, the following datasets can be downloaded from [OpenDataLab](https://opendatalab.com/) with high speed:

  - [MOT17](https://opendatalab.com/MOT17/download)
  - [CrowdHuman](https://opendatalab.com/CrowdHuman/download)
  - [LVIS](https://opendatalab.com/LVIS/download)
  - [TAO](https://opendatalab.com/TAO/download)

#### 1.2 Video Instance Segmentation

- For the training and testing of video instance segmetatioon task, only one of YouTube-VIS datasets (e.g. YouTube-VIS 2019) is needed.

- YouTube-VIS 2019 dataset can be download from [OpenDataLab](https://opendatalab.com/) (recommended for users in China): https://opendatalab.com/YouTubeVIS2019/download

#### 1.5 Data Structure

If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── annotations
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
│   │
|   ├── DanceTrack
|   |   ├── train
|   |   ├── val
|   |   ├── test
|   |
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │
│   ├── lvis
│   │   ├── train (the same as coco/train2017)
│   │   ├── val (the same as coco/val2017)
│   │   ├── test (the same as coco/test2017)
│   │   ├── annotations
│   │   │   ├── coco_to_lvis_synset.json
│   │   │   ├── lvis_v0.5_train.json
│   │   │   ├── lvis_v0.5_val.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_image_info_test_challenge.json
│   │   │   ├── lvis_v1_image_info_test_dev.json
│   │
│   ├── tao
│   │   ├── annotations
│   │   │   ├── test_without_annotations.json
│   │   │   ├── train.json
│   │   │   ├── validation.json
│   │   │   ├── ......
│   │   ├── test
│   │   │   ├── ArgoVerse
│   │   │   ├── AVA
│   │   │   ├── BDD
│   │   │   ├── Charades
│   │   │   ├── HACS
│   │   │   ├── LaSOT
│   │   │   ├── YFCC100M
│   │   ├── train
│   │   ├── val
│   │
│   ├── youtube_vis_2019
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── train.json (the official annotation files)
│   │   │── valid.json (the official annotation files)
│   │   │── test.json (the official annotation files)
│   │
│   ├── youtube_vis_2021
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
```

### 2. Convert Annotations

In this case, you need to convert the official annotations to coco style. We provide scripts and the usages are as following:

```shell

# MOT17
# The processing of other MOT Challenge dataset is the same as MOT17
python ./tools/dataset_converters/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/dataset_converters/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# DanceTrack
python ./tools/dataset_converters/dancetrack/dancetrack2coco.py -i ./data/DanceTrack ./data/DanceTrack/annotations

# CrowdHuman
python ./tools/dataset_converters/mot/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations

# LVIS
# Merge annotations from LVIS and COCO for training QDTrack
python ./tools/dataset_converters/tao/merge_coco_with_lvis.py --lvis ./data/lvis/annotations/lvis_v0.5_train.json --coco ./data/coco/annotations/instances_train2017.json --mapping ./data/lvis/annotations/coco_to_lvis_synset.json --output-json ./data/lvis/annotations/lvisv0.5+coco_train.json

# TAO
# Generate filtered json file for QDTrack
python ./tools/dataset_converters/tao/tao2coco.py -i ./data/tao/annotations --filter-classes

# YouTube-VIS 2019
python ./tools/dataset_converters/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2019 -o ./data/youtube_vis_2019/annotations --version 2019

# YouTube-VIS 2021
python ./tools/dataset_converters/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2021 -o ./data/youtube_vis_2021/annotations --version 2021
```

The folder structure will be as following after your run these scripts:

```
mmdetection
├── mmtrack
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── annotations
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
│   │   │   ├── imgs
│   │   │   ├── meta
│   │
│   ├── DanceTrack
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── annotations
│   │
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │   ├── annotations
│   │   │   ├── crowdhuman_train.json
│   │   │   ├── crowdhuman_val.json
│   │
│   ├── lvis
│   │   ├── train (the same as coco/train2017)
│   │   ├── val (the same as coco/val2017)
│   │   ├── test (the same as coco/test2017)
│   │   ├── annotations
│   │   │   ├── coco_to_lvis_synset.json
│   │   │   ├── lvisv0.5+coco_train.json
│   │   │   ├── lvis_v0.5_train.json
│   │   │   ├── lvis_v0.5_val.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_image_info_test_challenge.json
│   │   │   ├── lvis_v1_image_info_test_dev.json
│   │
│   ├── tao
│   │   ├── annotations
│   │   │   ├── test_482_classes.json
│   │   │   ├── test_without_annotations.json
│   │   │   ├── train.json
│   │   │   ├── train_482_classes.json
│   │   │   ├── validation.json
│   │   │   ├── validation_482_classes.json
│   │   │   ├── ......
│   │   ├── test
│   │   │   ├── ArgoVerse
│   │   │   ├── AVA
│   │   │   ├── BDD
│   │   │   ├── Charades
│   │   │   ├── HACS
│   │   │   ├── LaSOT
│   │   │   ├── YFCC100M
│   │   ├── train
│   │   ├── val
│   │
│   ├── youtube_vis_2019
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── train.json (the official annotation files)
│   │   │── valid.json (the official annotation files)
│   │   │── test.json (the official annotation files)
│   │   │── annotations (the converted annotation file)
│   │
│   ├── youtube_vis_2021
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── annotations (the converted annotation file)
```

#### The folder of annotations and reid in MOT15/MOT16/MOT17/MOT20

We take MOT17 dataset as examples, the other datasets share similar structure.

There are 8 JSON files in `data/MOT17/annotations`:

`train_cocoformat.json`: JSON file containing the annotations information of the training set in MOT17 dataset.

`train_detections.pkl`: Pickle file containing the public detections of the training set in MOT17 dataset.

`test_cocoformat.json`: JSON file containing the annotations information of the testing set in MOT17 dataset.

`test_detections.pkl`: Pickle file containing the public detections of the testing set in MOT17 dataset.

`half-train_cocoformat.json`, `half-train_detections.pkl`, `half-val_cocoformat.json`and `half-val_detections.pkl` share similar meaning with `train_cocoformat.json` and `train_detections.pkl`. The `half` means we split each video in the training set into half. The first half videos are denoted as `half-train` set, and the second half videos are denoted as`half-val` set.

The structure of `data/MOT17/reid` is as follows:

```
reid
├── imgs
│   ├── MOT17-02-FRCNN_000002
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
│   ├── MOT17-02-FRCNN_000003
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
├── meta
│   ├── train_80.txt
│   ├── val_20.txt
```

The `80` in `train_80.txt` means the proportion of the training dataset to the whole ReID dataset is 80%. While the proportion of the validation dataset is 20%.

For training, we provide a annotation list `train_80.txt`. Each line of the list contains a filename and its corresponding ground-truth labels. The format is as follows:

```
MOT17-05-FRCNN_000110/000018.jpg 0
MOT17-13-FRCNN_000146/000014.jpg 1
MOT17-05-FRCNN_000088/000004.jpg 2
MOT17-02-FRCNN_000009/000081.jpg 3
```

`MOT17-05-FRCNN_000110` denotes the 110-th person in `MOT17-05-FRCNN` video.

For validation, The annotation list `val_20.txt` remains the same as format above.

Images in `reid/imgs` are cropped from raw images in `MOT17/train` by the corresponding `gt.txt`. The value of ground-truth labels should fall in range `[0, num_classes - 1]`.

#### The folder of annotations in crowdhuman

There are 2 JSON files in `data/crowdhuman/annotations`:

`crowdhuman_train.json`:  JSON file containing the annotations information of the training set in CrowdHuman dataset.
`crowdhuman_val.json`:  JSON file containing the annotations information of the validation set in CrowdHuman dataset.

#### The folder of annotations in lvis

There are 8 JSON files in `data/lvis/annotations`

`coco_to_lvis_synset.json`: JSON file containing the mapping relationship between COCO and LVIS categories.

`lvisv0.5+coco_train.json`: JSON file containing the merged annotations.

`lvis_v0.5_train.json`: JSON file containing the annotations information of the training set in lvisv0.5.

`lvis_v0.5_val.json`: JSON file containing the annotations information of the validation set in lvisv0.5.

`lvis_v1_train.json`: JSON file containing the annotations information of the training set in lvisv1.

`lvis_v1_val.json`: JSON file containing the annotations information of the validation set in lvisv1.

`lvis_v1_image_info_test_challenge.json`: JSON file containing the annotations information of the testing set in lvisv1 available for year-round evaluation.

`lvis_v1_image_info_test_dev.json`: JSON file containing the annotations information of the testing set in lvisv1 available only once a year for LVIS Challenge.

#### The folder of annotations in tao

There are 9 JSON files in `data/tao/annotations`:

`test_categories.json`: JSON file containing a list of categories which will be evaluated on the TAO test set.

`test_without_annotations.json`:  JSON for test videos. The 'images' and 'videos' fields contain the images and videos that will be evaluated on the test set.

`test_482_classes.json`: JSON file containing the converted results for test set.

`train.json`: JSON file containing annotations for LVIS categories in TAO train.

`train_482_classes.json`: JSON file containing the converted results for train set.

`train_with_freeform.json`: JSON file containing annotations for all categories in TAO train.

`validation.json`: JSON file containing annotations for LVIS categories in TAO train.

`validation_482_classes.json`: JSON file containing the converted results for validation set.

`validation_with_freeform.json`: JSON file containing annotations for all categories in TAO validation.

#### The folder of annotations in youtube_vis_2019/youtube_vis2021

There are 3 JSON files in `data/youtube_vis_2019/annotations` or `data/youtube_vis_2021/annotations`:

`youtube_vis_2019_train.json`/`youtube_vis_2021_train.json`: JSON file containing the annotations information of the training set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_valid.json`/`youtube_vis_2021_valid.json`: JSON file containing the annotations information of the validation set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_test.json`/`youtube_vis_2021_test.json`: JSON file containing the annotations information of the testing set in youtube_vis_2019/youtube_vis2021 dataset.
