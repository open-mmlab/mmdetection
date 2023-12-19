# 数据准备和处理

## MM-GDINO-T 预训练数据准备和处理

MM-GDINO-T 模型中我们一共提供了 5 种不同数据组合的预训练配置，数据采用逐步累加的方式进行训练，因此用户可以根据自己的实际需求准备数据。

### 1 Object365 v1

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
python tools/dataset_converters/goldg2odvg.py data/gqa/final_mixed_train_no_coco.json
```

程序运行完成后会在 `data/gqa` 目录下创建 `final_flickr_separateGT_train_vg.json` 新文件，完整结构如下：

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



### 3 ODinW
### 4 D-CUBE
### 5 Flickr30k
### 6 Referring Expression Comprehension

## 微调数据集准备
### 1 COCO
### 2 LVIS 1.0 
### 3 RTTS
### 4 RUOD
### 5 Brain Tumor
### 6 Cityscapes
### 7 People in Painting
### 8 Referring Expression Comprehension


