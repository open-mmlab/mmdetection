# Training a ReID Model

You may want to train a ReID model for multiple object tracking or other applications. We support ReID model training in MMDetection, which is built upon [MMClassification](https://github.com/open-mmlab/mmclassification).

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Preparation

This section will show how to train a ReID model on standard datasets i.e. MOT17.

We need to download datasets following docs. We use [ReIDDataset](mmdet/datasets/reid_dataset.py) to maintain standard datasets. In this case, you need to convert the official dataset to this style. We provide scripts and the usages as follow:

```python
python tools/dataset_converters/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3
```

Arguments:

- `--val-split`: Proportion of the validation dataset to the whole ReID dataset.
- `--vis-threshold`: Threshold of visibility for each person.

The directory of the converted datasets is as follows:

```
MOT17
├── train
├── test
├── reid
│   ├── imgs
│   │   ├── MOT17-02-FRCNN_000002
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   │   ├── MOT17-02-FRCNN_000003
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   ├── meta
│   │   ├── train_80.txt
│   │   ├── val_20.txt
```

Note: `80` in `train_80.txt` means the proportion of the training dataset to the whole ReID dataset is eighty percent. While the proportion of the validation dataset is twenty percent.

For training, we provide a annotation list `train_80.txt`. Each line of the list constraints a filename and its corresponding ground-truth labels. The format is as follows:

```
MOT17-05-FRCNN_000110/000018.jpg 0
MOT17-13-FRCNN_000146/000014.jpg 1
MOT17-05-FRCNN_000088/000004.jpg 2
MOT17-02-FRCNN_000009/000081.jpg 3
```

For validation, The annotation list `val_20.txt` remains the same as format above.

Note: Images in `MOT17/reid/imgs` are cropped from raw images in `MOT17/train` by the corresponding `gt.txt`. The value of ground-truth labels should fall in range `[0, num_classes - 1]`.

### 3. Training

#### Training on a single GPU

```shell
python tools/train.py configs/reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py
```

#### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash tools/dist_train.sh configs/reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py 8
```

### 4. Customize Dataset

This section will show how to train a ReID model on customize datasets.

### 4.1 Dataset Preparation

You need to convert your customize datasets to existing dataset format.

#### An example of customized dataset

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The directory of the dataset is as follows:

```
Filelist
├── imgs
│   ├── person1
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
│   ├── person2
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
├── meta
│   ├── train.txt
│   ├── val.txt
```

The format of annotation list is as follows:

```
person1/000000.jpg 0
person1/000001.jpg 0
person2/000000.jpg 1
person2/000001.jpg 1
```

You can directly use [ReIDDataset](mmdet/datasets/reid_dataset.py). In this case, you only need to modify the config as follows:

```python
# modify the path of annotation files and the image path prefix
data = dict(
    train=dict(
        data_prefix='data/Filelist/imgs',
        ann_file='data/Filelist/meta/train.txt'),
    val=dict(
        data_prefix='data/Filelist/imgs',
        ann_file='data/Filelist/meta/val.txt'),
    test=dict(
        data_prefix='data/Filelist/imgs',
        ann_file='data/Filelist/meta/val.txt'),
)
# modify the number of classes, assume your training set has 100 classes
model = dict(reid=dict(head=dict(num_classes=100)))
```

### 4.2 Training

The training stage is the same as `Standard Dataset`.
