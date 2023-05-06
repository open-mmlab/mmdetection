## Dataset Preparation

This page provides the instructions for dataset preparation on existing benchmarks, include

- Multiple Object Tracking
  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)

### 1. Download Datasets

Please download the datasets from the official websites. It is recommended to symlink the root of the datasets to `$MMDETECTION/data`.

#### 1.1 Multiple Object Tracking

- For the training and testing of multi object tracking task, MOT17 is needed, CrowdHuman can be served as comlementary dataset.

- For users in China, the following datasets can be downloaded from [OpenDataLab](https://opendatalab.com/) with high speed:

  - [MOT17](https://opendatalab.com/MOT17/download)
  - [CrowdHuman](https://opendatalab.com/CrowdHuman/download)

#### 1.2 Data Structure

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
```

### 2. Convert Annotations

In this case, you need to convert the official annotations to coco style. We provide scripts and the usages are as following:

```shell
# MOT17
# The processing of other MOT Challenge dataset is the same as MOT17
python ./tools/dataset_converters/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/dataset_converters/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# CrowdHuman
python ./tools/dataset_converters/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations

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
