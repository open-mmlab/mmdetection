## 数据集准备

本页面提供了现有基准数据集的准备说明，包括：

- 多目标跟踪

  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)

- 视频实例分割

  - [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### 1. 下载数据集

请从官方网站下载数据集，并将数据集的根目录建立软链接到 `$MMDETECTION/data` 目录下。

#### 1.1 多目标跟踪

- 对于多目标跟踪任务的训练和测试，需要下载MOT Challenge数据集之一（例如MOT17、MOT20），CrowdHuman数据集可以作为补充数据集。

- 对于中国的用户，可以从 [OpenDataLab](https://opendatalab.com/) 上高速下载如下数据集：

  - [MOT17](https://opendatalab.com/MOT17/download)
  - [MOT20](https://opendatalab.com/MOT20/download)
  - [CrowdHuman](https://opendatalab.com/CrowdHuman/download)

#### 1.2 视频实例分割

- 对于视频实例分割任务的训练和测试，只需要选择一个YouTube-VIS数据集（例如YouTube-VIS 2019、YouTube-VIS 2021）即可。
- 可以从 [YouTubeVOS](https://codalab.lisn.upsaclay.fr/competitions/6064) 上下载YouTube-VIS 2019数据集。
- 可以从 [YouTubeVOS](https://codalab.lisn.upsaclay.fr/competitions/7680) 上下载YouTube-VIS 2021数据集。

#### 1.3 数据结构

如果您的文件夹结构与以下结构不同，则可能需要在配置文件中更改相应的路径。

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
|   |   |   ├── MOT17-02-DPM
|   |   |   |   ├── det
|   │   │   │   ├── gt
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
|   |   ├── test
|   |   |   ├── MOT17-01-DPM
|   |   |   |   ├── det
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
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

### 2. 转换注释

在这种情况下，您需要将官方注释（Annotations）转换为COCO格式。我们提供了相应的脚本，使用方法如下：

```shell
# MOT17
# 其他 MOT Challenge 数据集的处理方式与 MOT17 相同。
python ./tools/dataset_converters/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/dataset_converters/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# CrowdHuman
python ./tools/dataset_converters/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations

# YouTube-VIS 2019
python ./tools/dataset_converters/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2019 -o ./data/youtube_vis_2019/annotations --version 2019

# YouTube-VIS 2021
python ./tools/dataset_converters/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2021 -o ./data/youtube_vis_2021/annotations --version 2021

```

运行这些脚本后，文件夹结构将如下所示：

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
|   |   |   ├── MOT17-02-DPM
|   |   |   |   ├── det
|   │   │   │   ├── gt
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
|   |   ├── test
|   |   |   ├── MOT17-01-DPM
|   |   |   |   ├── det
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
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

#### MOT15/MOT16/MOT17/MOT20中的注释和reid文件夹

以 MOT17 数据集为例，其他数据集的结构类似。

在 `data/MOT17/annotations` 文件夹中有8个JSON文件：

`train_cocoformat.json`: 包含MOT17数据集训练集的注释信息的JSON文件。

`train_detections.pkl`: 包含MOT17数据集训练集的公共检测结果的Pickle文件。

`test_cocoformat.json`: 包含MOT17数据集测试集的注释信息的JSON文件。

`test_detections.pkl`: 包含MOT17数据集测试集的公共检测结果的Pickle文件。

`half-train_cocoformat.json`、`half-train_detections.pkl`、`half-val_cocoformat.json` 和 `half-val_detections.pkl` 与 `train_cocoformat.json` 和 `train_detections.pkl` 具有类似的含义。`half` 表示将训练集中的每个视频分成两半。前一半的视频被标记为 `half-train` 集，后一半的视频被标记为 `half-val` 集。

`data/MOT17/reid` 文件夹的结构如下所示：

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

`train_80.txt` 中的 `80` 表示训练数据集在整个ReID数据集中的比例为80%。而验证数据集的比例为20%。

关于训练，我们提供了一个注释列表 `train_80.txt`。列表中的每一行包含一个文件名及其对应的真实标签。格式如下所示：

```
MOT17-05-FRCNN_000110/000018.jpg 0
MOT17-13-FRCNN_000146/000014.jpg 1
MOT17-05-FRCNN_000088/000004.jpg 2
MOT17-02-FRCNN_000009/000081.jpg 3
```

`MOT17-05-FRCNN_000110` 表示 `MOT17-05-FRCNN` 视频中的第110个人。

对于验证集，注释列表 `val_20.txt` 的格式与上述相同。

`reid/imgs` 中的图像是通过相应的 `gt.txt` 从 `MOT17/train` 中的原始图像中裁剪而来。真实标签的值应在 `[0, num_classes - 1]` 的范围内。

#### CrowdHuman 中的 annotations 文件夹

`data/crowdhuman/annotations` 文件夹下有两个JSON文件：

`crowdhuman_train.json`：包含 CrowdHuman 数据集训练集的注释信息的JSON文件。
`crowdhuman_val.json`：包含 CrowdHuman 数据集验证集的注释信息的JSON文件。

#### youtube_vis_2019/youtube_vis2021 中的 annotations 文件夹

There are 3 JSON files in `data/youtube_vis_2019/annotations` or `data/youtube_vis_2021/annotations`：

`youtube_vis_2019_train.json`/`youtube_vis_2021_train.json`：包含 youtube_vis_2019/youtube_vis2021 数据集训练集的注释信息的JSON文件。

`youtube_vis_2019_valid.json`/`youtube_vis_2021_valid.json`：包含 youtube_vis_2019/youtube_vis2021 数据集验证集的注释信息的JSON文件。

`youtube_vis_2019_test.json`/`youtube_vis_2021_test.json`：包含 youtube_vis_2019/youtube_vis2021 数据集测试集的注释信息的JSON文件。
