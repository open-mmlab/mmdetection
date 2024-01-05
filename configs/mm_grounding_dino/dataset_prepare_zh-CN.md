# 数据准备和处理

## MM-GDINO-T 预训练数据准备和处理

MM-GDINO-T 模型中我们一共提供了 5 种不同数据组合的预训练配置，数据采用逐步累加的方式进行训练，因此用户可以根据自己的实际需求准备数据。

### 1 Objects365 v1

对应的训练配置为 [grounding_dino_swin-t_pretrain_obj365](./grounding_dino_swin-t_pretrain_obj365.py)

Objects365_v1 可以从 [opendatalab](https://opendatalab.com/OpenDataLab/Objects365_v1) 下载，其提供了 CLI 和 SDK 两者下载方式。

下载并解压后，将其放置或者软链接到 `data/objects365v1` 目录下，目录结构如下：

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

然后使用 [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
```

程序运行完成后会在 `data/objects365v1` 目录下创建 `o365v1_train_od.json` 和 `o365v1_label_map.json` 两个新文件，完整结构如下：

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

上述配置在训练过程中会评估 COCO 2017 数据集的性能，因此需要准备 COCO 2017 数据集。你可以从 [COCO](https://cocodataset.org/) 官网下载或者从 [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017) 下载

下载并解压后，将其放置或者软链接到 `data/coco` 目录下，目录结构如下：

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

下载该数据集后就可以训练 [grounding_dino_swin-t_pretrain_obj365_goldg](./grounding_dino_swin-t_pretrain_obj365_goldg.py) 配置了。

GoldG 数据集包括 `GQA` 和 `Flickr30k` 两个数据集，来自 GLIP 论文中提到的 MixedGrounding 数据集，其排除了 COCO 数据集。下载链接为 [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations)，我们目前需要的是 `mdetr_annotations/final_mixed_train_no_coco.json` 和 `mdetr_annotations/final_flickr_separateGT_train.json` 文件。

然后下载 [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) 图片。下载并解压后，将其放置或者软链接到 `data/gqa` 目录下，目录结构如下：

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

然后下载 [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/) 图片。这个数据下载需要先申请，再获得下载链接后才可以下载。下载并解压后，将其放置或者软链接到 `data/flickr30k_entities` 目录下，目录结构如下：

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

对于 GQA 数据集，你需要使用 [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/goldg2odvg.py data/gqa/final_mixed_train_no_coco.json
```

程序运行完成后会在 `data/gqa` 目录下创建 `final_mixed_train_no_coco_vg.json` 新文件，完整结构如下：

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

对于 Flickr30k 数据集，你需要使用 [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/goldg2odvg.py data/flickr30k_entities/final_flickr_separateGT_train.json
```

程序运行完成后会在 `data/flickr30k_entities` 目录下创建 `final_flickr_separateGT_train_vg.json` 新文件，完整结构如下：

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

对应的训练配置为 [grounding_dino_swin-t_pretrain_obj365_goldg_grit9m](./grounding_dino_swin-t_pretrain_obj365_goldg_grit9m.py)

GRIT数据集可以从 [GRIT](https://huggingface.co/datasets/zzliang/GRIT#download-image) 中使用 img2dataset 包下载，默认指令下载后数据集大小为 1.1T，下载和处理预估需要至少 2T 硬盘空间，可根据硬盘容量酌情下载。下载后原始格式为：

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

下载后需要对格式进行进一步处理:

```shell
python tools/dataset_converters/grit_processing.py data/grit_raw data/grit_processed
```

处理后的格式为：

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

对于 GRIT 数据集，你需要使用 [grit2odvg.py](../../tools/dataset_converters/grit2odvg.py) 转化成需要的 ODVG 格式：

```shell
python tools/dataset_converters/grit2odvg.py data/grit_processed/
```

程序运行完成后会在 `data/grit_processed` 目录下创建 `grit20m_vg.json` 新文件，大概包含 9M 条数据，完整结构如下：

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

对应的训练配置为

- [grounding_dino_swin-t_pretrain_obj365_goldg_v3det](./grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py)
- [grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det](./grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py)

V3Det 数据集下载可以从 [opendatalab](https://opendatalab.com/V3Det/V3Det) 下载，下载并解压后，将其放置或者软链接到 `data/v3det` 目录下，目录结构如下：

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

然后使用 [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/coco2odvg.py data/v3det/annotations/v3det_2023_v1_train.json -d v3det
```

程序运行完成后会在 `data/v3det/annotations` 目录下创建目录下创建 `v3det_2023_v1_train_od.json` 和 `v3det_2023_v1_label_map.json` 两个新文件，完整结构如下：

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

### 6 数据切分和可视化

考虑到用户需要准备的数据集过多，不方便对图片和标注进行训练前确认，因此我们提供了一个数据切分和可视化的工具，可以将数据集切分为 tiny 版本，然后使用可视化脚本查看图片和标签正确性。

1. 切分数据集

脚本位于 [这里](../../tools/misc/split_odvg.py), 以 `Object365 v1` 为例，切分数据集的命令如下：

```shell
python tools/misc/split_odvg.py data/object365_v1/ o365v1_train_od.json train your_output_dir --label-map-file o365v1_label_map.json -n 200
```

上述脚本运行后会在 `your_output_dir` 目录下创建和 `data/object365_v1/` 一样的文件夹结构，但是只会保存 200 张训练图片和对应的 json，方便用户查看。

2. 可视化原始数据集

脚本位于 [这里](../../tools/analysis_tools/browse_grounding_raw.py), 以 `Object365 v1` 为例，可视化数据集的命令如下：

```shell
python tools/analysis_tools/browse_grounding_raw.py data/object365_v1/ o365v1_train_od.json train --label-map-file o365v1_label_map.json -o your_output_dir --not-show
```

上述脚本运行后会在 `your_output_dir` 目录下生成同时包括图片和标签的图片，方便用户查看。

3. 可视化 dataset 输出的数据集

脚本位于 [这里](../../tools/analysis_tools/browse_grounding_dataset.py), 用户可以通过该脚本查看 dataset 输出的结果即包括了数据增强的结果。 以 `Object365 v1` 为例，可视化数据集的命令如下：

```shell
python tools/analysis_tools/browse_grounding_dataset.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py  -o your_output_dir --not-show
```

上述脚本运行后会在 `your_output_dir` 目录下生成同时包括图片和标签的图片，方便用户查看。

## MM-GDINO-L 预训练数据准备和处理

### 1 Object365 v2

Objects365_v2 可以从 [opendatalab](https://opendatalab.com/OpenDataLab/Objects365) 下载，其提供了 CLI 和 SDK 两者下载方式。

下载并解压后，将其放置或者软链接到 `data/objects365v2` 目录下，目录结构如下：

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

由于 objects365v2 类别中有部分类名是错误的，因此需要先进行修正。

```shell
python tools/dataset_converters/fix_o365_names.py
```

会在 `data/objects365v2/annotations` 下生成新的标注文件 `zhiyuan_objv2_train_fixname.json`。

然后使用 [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v2/annotations/zhiyuan_objv2_train_fixname.json -d o365v2
```

程序运行完成后会在 `data/objects365v2` 目录下创建 `zhiyuan_objv2_train_fixname_od.json` 和 `o365v2_label_map.json` 两个新文件，完整结构如下：

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

OpenImages v6 可以从 [官网](https://storage.googleapis.com/openimages/web/download_v6.html) 下载，由于数据集比较大，需要花费一定的时间，下载完成后文件结构如下：

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

然后使用 [openimages2odvg.py](../../tools/dataset_converters/openimages2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/openimages2odvg.py data/OpenImages/annotations
```

程序运行完成后会在 `data/OpenImages/annotations` 目录下创建 `oidv6-train-annotation_od.json` 和 `openimages_label_map.json` 两个新文件，完整结构如下：

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

参见前面的 MM-GDINO-T 预训练数据准备和处理 数据准备部分，完整数据集结构如下：

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

参见后面的 `微调数据集准备` 的 `2 LVIS 1.0` 部分。完整数据集结构如下：

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

数据准备可以参考前面的 `MM-GDINO-T 预训练数据准备和处理` 部分。为了方便后续处理，请将下载的 [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations) 文件夹软链接或者移动到 `data/coco` 路径下
完整数据集结构如下：

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

由于 COCO2017 train 和 RefCOCO/RefCOCO+/RefCOCOg/gRefCOCO val 中存在部分重叠，如果不提前移除，在评测 RefExp 时候会存在数据泄露。

```shell
python tools/dataset_converters/remove_cocotrain2017_from_refcoco.py data/coco/mdetr_annotations data/coco/annotations/instances_train2017.json
```

会在 `data/coco/annotations` 目录下创建 `instances_train2017_norefval.json` 新文件。最后使用 [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/coco2odvg.py data/coco/annotations/instances_train2017_norefval.json -d coco
```

会在 `data/coco/annotations` 目录下创建 `instances_train2017_norefval_od.json` 和 `coco_label_map.json` 两个新文件，完整结构如下：

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

注意： COCO2017 train 和 LVIS 1.0 val 数据集有 15000 张图片重复，因此一旦在训练中使用了 COCO2017 train，那么 LVIS 1.0 val 的评测结果就存在数据泄露问题，LVIS 1.0 minival 没有这个问题。

### 6 GoldG

参见 MM-GDINO-T 预训练数据准备和处理 部分

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

MDetr 中提供了 COCO2014 train 的 Phrase Grounding 版本标注， 最原始标注文件为 `final_mixed_train.json`，和之前类似，文件结构如下：

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

我们可以从 `final_mixed_train.json` 中提取出 COCO 部分数据

```shell
python tools/dataset_converters/extract_coco_from_mixed.py data/coco/mdetr_annotations/final_mixed_train.json
```

会在 `data/coco/mdetr_annotations` 目录下创建 `final_mixed_train_only_coco.json` 新文件，最后使用 [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/goldg2odvg.py data/coco/mdetr_annotations/final_mixed_train_only_coco.json
```

会在 `data/coco/mdetr_annotations` 目录下创建 `final_mixed_train_only_coco_vg.json` 新文件，完整结构如下：

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

注意： COCO2014 train 和 COCO2017 val 没有重复图片，因此不用担心 COCO 评测的数据泄露问题。

### 8 Referring Expression Comprehension

其一共包括 4 个数据集。数据准备部分请参见 微调数据集准备 部分。

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

参见 MM-GDINO-T 预训练数据准备和处理 部分

## 评测数据集准备

### 1 COCO 2017

数据准备流程和前面描述一致，最终结构如下：

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

LVIS 1.0 val 数据集包括 mini 和全量两个版本，mini 版本存在的意义是：

1. LVIS val 全量评测数据集比较大，评测一次需要比较久的时间
2. LVIS val 全量数据集中包括了 15000 张 COCO2017 train, 如果用户使用了 COCO2017 数据进行训练，那么将存在数据泄露问题

LVIS 1.0 图片和 COCO2017 数据集图片完全一样，只是提供了新的标注而已，minival 标注文件可以从 [这里](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json)下载， val 1.0 标注文件可以从 [这里](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_od_val.json) 下载。 最终结构如下：

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

ODinw 全称为 Object Detection in the Wild，是用于验证 grounding 预训练模型在不同实际场景中的泛化能力的数据集，其包括两个子集，分别是 ODinW13 和 ODinW35，代表是由 13 和 35 个数据集组成的。你可以从 [这里](https://huggingface.co/GLIPModel/GLIP/tree/main/odinw_35)下载，然后对每个文件进行解压，最终结构如下：

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

在评测 ODinW3535 时候由于需要自定义 prompt，因此需要提前对标注的 json 文件进行处理，你可以使用 [override_category.py](./odinw/override_category.py) 脚本进行处理，处理后会生成新的标注文件，不会覆盖原先的标注文件。

```shell
python configs/mm_grounding_dino/odinw/override_category.py data/odinw/
```

### 4 DOD

DOD 来自 [Described Object Detection: Liberating Object Detection with Flexible Expressions](https://arxiv.org/abs/2307.12813)。其数据集可以从 [这里](https://github.com/shikras/d-cube?tab=readme-ov-file#download)下载，最终的数据集结构如下：

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

在前面 GoldG 数据准备章节中我们已经下载了 Flickr30k 训练所需文件，评估所需的文件是 2 个 json 文件，你可以从 [这里](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_val.json) 和 [这里](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_test.json)下载，最终的数据集结构如下：

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

指代性表达式理解包括 4 个数据集： RefCOCO, RefCOCO+, RefCOCOg, gRefCOCO。这 4 个数据集所采用的图片都来自于 COCO2014 train，和 COCO2017 类似，你可以从 COCO 官方或者 opendatalab 中下载，而标注可以直接从 [这里](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations) 下载，mdetr_annotations 文件夹里面包括了其他大量的标注，你如果觉得数量过多，可以只下载所需要的几个 json 文件即可。最终的数据集结构如下：

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

注意 gRefCOCO 是在 [GREC: Generalized Referring Expression Comprehension](https://arxiv.org/abs/2308.16182) 被提出，并不在 `mdetr_annotations` 文件夹中，需要自行处理。具体步骤为：

1. 下载 [gRefCOCO](https://github.com/henghuiding/gRefCOCO?tab=readme-ov-file#grefcoco-dataset-download)，并解压到 data/coco/ 文件夹中

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

2. 转换为 coco 格式

你可以使用 gRefCOCO 官方提供的[转换脚本](https://github.com/henghuiding/gRefCOCO/blob/b4b1e55b4d3a41df26d6b7d843ea011d581127d4/mdetr/scripts/fine-tuning/grefexp_coco_format.py)。注意需要将被注释的 161 行打开，并注释 160 行才可以得到全量的 json 文件。

```shell
# 需要克隆官方 repo
git clone https://github.com/henghuiding/gRefCOCO.git
cd gRefCOCO/mdetr
python scripts/fine-tuning/grefexp_coco_format.py --data_path ../../data/coco/grefs --out_path ../../data/coco/mdetr_annotations/ --coco_path ../../data/coco
```

会在 `data/coco/mdetr_annotations/` 文件夹中生成 4 个 json 文件，完整的数据集结构如下：

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

## 微调数据集准备

### 1 COCO 2017

COCO 是检测领域最常用的数据集，我们希望能够更充分探索其微调模式。从目前发展来看，一共有 3 种微调方式：

1. 闭集微调，即微调后文本端将无法修改描述，转变为闭集算法，在 COCO 上性能能够最大化，但是失去了通用性。
2. 开集继续预训练微调，即对 COCO 数据集采用和预训练一致的预训练手段。此时有两种做法，第一种是降低学习率并固定某些模块，仅仅在 COCO 数据上预训练，第二种是将 COCO 数据和部分预训练数据混合一起训练，两种方式的目的都是在尽可能不降低泛化性时提高 COCO 数据集性能
3. 开放词汇微调，即采用 OVD 领域常用做法，将 COCO 类别分成 base 类和 novel 类，训练时候仅仅在 base 类上进行，评测在 base 和 novel 类上进行。这种方式可以验证 COCO OVD 能力，目的也是在尽可能不降低泛化性时提高 COCO 数据集性能

**(1) 闭集微调**

这个部分无需准备数据，直接用之前的数据即可。

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

**(2) 开集继续预训练微调**
这种方式需要将 COCO 训练数据转换为 ODVG 格式，你可以使用如下命令转换：

```shell
python tools/dataset_converters/coco2odvg.py data/coco/annotations/instances_train2017.json -d coco
```

会在 `data/coco/annotations/` 下生成新的 `instances_train2017_od.json` 和 `coco2017_label_map.json`，完整的数据集结构如下：

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

在得到数据后，你可以自行选择单独预习还是混合预训练方式。

**(3) 开放词汇微调**
这种方式需要将 COCO 训练数据转换为 OVD 格式，你可以使用如下命令转换：

```shell
python tools/dataset_converters/coco2ovd.py data/coco/
```

会在 `data/coco/annotations/` 下生成新的 `instances_val2017_all_2.json` 和 `instances_val2017_seen_2.json`，完整的数据集结构如下：

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

然后可以直接使用 [配置](coco/grounding_dino_swin-t_finetune_16xb4_1x_coco_48_17.py) 进行训练和测试。

### 2 LVIS 1.0

LVIS 是一个包括 1203 类的数据集，同时也是一个长尾联邦数据集，对其进行微调很有意义。 由于其类别过多，我们无法对其进行闭集微调，因此只能采用开集继续预训练微调和开放词汇微调。

你需要先准备好 LVIS 训练 JSON 文件，你可以从 [这里](https://www.lvisdataset.org/dataset) 下载，我们只需要 `lvis_v1_train.json` 和 `lvis_v1_val.json`，然后将其放到 `data/coco/annotations/` 下，然后运行如下命令：

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

(1) 开集继续预训练微调

使用如下命令转换为 ODVG 格式：

```shell
python tools/dataset_converters/lvis2odvg.py data/coco/annotations/lvis_v1_train.json
```

会在 `data/coco/annotations/` 下生成新的 `lvis_v1_train_od.json` 和 `lvis_v1_label_map.json`，完整的数据集结构如下：

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

然后可以直接使用 [配置](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis.py) 进行训练测试，或者你修改配置将其和部分预训练数据集混合使用。

**(2) 开放词汇微调**

使用如下命令转换为 OVD 格式：

```shell
python tools/dataset_converters/lvis2ovd.py data/coco/
```

会在 `data/coco/annotations/` 下生成新的 `lvis_v1_train_od_norare.json` 和 `lvis_v1_label_map_norare.json`，完整的数据集结构如下：

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

然后可以直接使用 [配置](lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py) 进行训练测试

### 3 RTTS

RTTS 是一个浓雾天气数据集，该数据集包含 4,322 张雾天图像，包含五个类：自行车 (bicycle)、公共汽车 (bus)、汽车 (car)、摩托车 (motorbike) 和人 (person)。可以从 [这里](https://drive.google.com/file/d/15Ei1cHGVqR1mXFep43BO7nkHq1IEGh1e/view)下载, 然后解压到 `data/RTTS/` 文件夹中。完整的数据集结构如下：

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

RUOD 是一个水下目标检测数据集，你可以从 [这里](https://drive.google.com/file/d/1hxtbdgfVveUm_DJk5QXkNLokSCTa_E5o/view)下载, 然后解压到 `data/RUOD/` 文件夹中。完整的数据集结构如下：

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

Brain Tumor 是一个医学领域的 2d 检测数据集，你可以从 [这里](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp/dataset/2)下载, 请注意选择 `COCO JSON` 格式。然后解压到 `data/brain_tumor_v2/` 文件夹中。完整的数据集结构如下：

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

Cityscapes 是一个城市街景数据集，你可以从 [这里](https://www.cityscapes-dataset.com/) 或者 opendatalab 中下载, 然后解压到 `data/cityscapes/` 文件夹中。完整的数据集结构如下：

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

在下载后，然后使用 [cityscapes.py](../../tools/dataset_converters/cityscapes.py) 脚本生成我们所需要的 json 格式

```shell
python tools/dataset_converters/cityscapes.py data/cityscapes/
```

会在 annotations 中生成 3 个新的 json 文件。完整的数据集结构如下：

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

People in Painting 是一个油画数据集，你可以从 [这里](https://universe.roboflow.com/roboflow-100/people-in-paintings/dataset/2), 请注意选择 `COCO JSON` 格式。然后解压到 `data/people_in_painting_v2/` 文件夹中。完整的数据集结构如下：

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

指代性表达式理解的微调和前面一样，也是包括 4 个数据集，在评测数据准备阶段已经全部整理好了，完整的数据集结构如下：

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

然后我们需要将其转换为所需的 ODVG 格式，请使用 [refcoco2odvg.py](../../tools/dataset_converters/refcoco2odvg.py) 脚本转换，

```shell
python tools/dataset_converters/refcoco2odvg.py data/coco/mdetr_annotations
```

会在 `data/coco/mdetr_annotations` 中生成新的 4 个 json 文件。 转换后的数据集结构如下：

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
