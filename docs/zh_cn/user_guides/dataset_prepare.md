## 数据集准备

### 基础检测数据集准备

MMDetection 支持多个公共数据集，包括 [COCO](https://cocodataset.org/)， [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)， [Cityscapes](https://www.cityscapes-dataset.com/) 和 [其他更多数据集](https://github.com/open-mmlab/mmdetection/tree/main/configs/_base_/datasets)。

一些公共数据集，比如 Pascal VOC 及其镜像数据集，或者 COCO 等数据集都可以从官方网站或者镜像网站获取。注意：在检测任务中，Pascal VOC 2012 是 Pascal VOC 2007 的无交集扩展，我们通常将两者一起使用。 我们建议将数据集下载，然后解压到项目外部的某个文件夹内，然后通过符号链接的方式，将数据集根目录链接到 `$MMDETECTION/data` 文件夹下， 如果你的文件夹结构和下方不同的话，你需要在配置文件中改变对应的路径。

我们提供了下载 COCO 等数据集的脚本，你可以运行 `python tools/misc/download_dataset.py --dataset-name coco2017` 下载 COCO 数据集。 对于中国境内的用户，我们也推荐通过开源数据平台 [OpenDataLab](https://opendatalab.com/?source=OpenMMLab%20GitHub) 来下载数据，以获得更好的下载体验。

更多用法请参考[数据集下载](./useful_tools.md#dataset-download)

```text
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

有些模型需要额外的 [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) 数据集，比如 HTC，DetectoRS 和 SCNet，你可以下载并解压它们到 `coco` 文件夹下。文件夹会是如下结构：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── stuffthingmaps
```

PanopticFPN 等全景分割模型需要额外的 [COCO Panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) 数据集，你可以下载并解压它们到 `coco/annotations` 文件夹下。文件夹会是如下结构：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

Cityscape 数据集的标注格式需要转换，以与 COCO 数据集标注格式保持一致，使用 `tools/dataset_converters/cityscapes.py` 来完成转换：

```shell
pip install cityscapesscripts

python tools/dataset_converters/cityscapes.py \
    ./data/cityscapes \
    --nproc 8 \
    --out-dir ./data/cityscapes/annotations
```

### COCO Caption 数据集准备

COCO Caption 采用的是 COCO2014 数据集作为图片，并且使用了 karpathy 的标注，

首先你需要下载 COCO2014 数据集

```shell
python tools/misc/download_dataset.py --dataset-name coco2014 --unzip
```

数据集会下载到当前路径的 `data/coco` 下。然后下载 karpathy 的标注

```shell
cd data/coco/annotations
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json
```

最终直接可用于训练和测试的数据集文件夹结构如下：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── coco_karpathy_train.json
│   │   │   ├── coco_karpathy_test.json
│   │   │   ├── coco_karpathy_val.json
│   │   │   ├── coco_karpathy_val_gt.json
│   │   │   ├── coco_karpathy_test_gt.json
│   │   ├── train2014
│   │   ├── val2014
│   │   ├── test2014
```

### COCO semantic 数据集准备

COCO 语义分割有两种类型标注，主要差别在于类别名定义不一样，因此处理方式也有两种，第一种是直接使用 stuffthingmaps 数据集，第二种是使用 panoptic 数据集。

**(1) 使用 stuffthingmaps 数据集**

该数据集的下载地址为 [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)，请下载后解压到 `data/coco` 文件夹下。

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── stuffthingmaps
```

该数据集不同于标准的 COCO 类别标注，其包括 172 个类： 80 thing 类、91 stuff 类和 1 个 'unlabeled'，其每个类别的说明见 https://github.com/nightrome/cocostuff/blob/master/labels.md

虽然只标注了 172 个类别，但是 `stuffthingmaps` 中最大标签 id 是 182，中间有些类别是没有标注的，并且第 0 类的 `unlabeled` 类别被移除。因此最终的 `stuffthingmaps` 图片中每个位置的值对应的类别关系见 https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/cocostuff/labels.txt

考虑到训练高效和方便用户，在开启训练或者评估前，我们需要将没有标注的 12 个类移除，这 12 个类的名字为： `street sign、hat、shoe、eye glasses、plate、mirror、window、desk、door、blender、hair brush`，最终可用于训练和评估的类别信息见 `mmdet/datasets/coco_semantic.py`

你可以使用 `tools/dataset_converters/coco_stuff164k.py` 来完成将下载的 `stuffthingmaps` 转换为直接可以训练和评估的数据集，转换后的数据集文件夹结构如下：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── stuffthingmaps
│   │   ├── stuffthingmaps_semseg
```

`stuffthingmaps_semseg` 即为新生成的可以直接训练和测试的 COCO 语义分割数据集。

**(2) 使用 panoptic 数据集**

通过 panoptic 标注生成的语义分割数据集类别数相比使用 `stuffthingmaps` 数据集生成的会少一些。首先你需要准备全景分割标注，然后使用如下脚本完成转换

```shell
python tools/dataset_converters/prepare_coco_semantic_annos_from_panoptic_annos.py data/coco
```

转换后的数据集文件夹结构如下：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   │   ├── panoptic_semseg_train2017
│   │   │   ├── panoptic_semseg_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

`panoptic_semseg_train2017` 和 `panoptic_semseg_val2017` 即为新生成的可以直接训练和测试的 COCO 语义分割数据集。注意其类别信息就是 COCO 全景分割的类别信息，包括 thing 和 stuff。

### RefCOCO 数据集准备

[RefCOCO](https://github.com/lichengunc/refer)系列数据集的图像和注释可以通过运行 `tools/misc/download_dataset.py` 下载:

```shell
python tools/misc/download_dataset.py --dataset-name refcoco --save-dir data/coco --unzip
```

然后，目录应该是这样的：

```text
data
├── coco
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   ├── refcocog
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(umd).p
|   |── train2014
```

### ADE20K 数据集准备

[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)数据集的图像和注释可以通过运行 `tools/misc/download_dataset.py` 下载:

```shell
python tools/misc/download_dataset.py --dataset-name ade20k_2016 --save-dir data --unzip
```

然后将注释移至`data/ADEChallengeData2016`目录，并运行预处理脚本以产生coco格式注释：

```shell
mv data/annotations_instance data/ADEChallengeData2016/
mv data/categoryMapping.txt data/ADEChallengeData2016/
mv data/imgCatIds.json data/ADEChallengeData2016/
python tools/dataset_converters/ade20k2coco.py data/ADEChallengeData2016 --task panoptic
python tools/dataset_converters/ade20k2coco.py data/ADEChallengeData2016 --task instance
```

然后，目录应该是这样的：

```text
data
├── ADEChallengeData2016
│   ├── ade20k_instance_train.json
│   ├── ade20k_instance_val.json
│   ├── ade20k_panoptic_train
|   |   ├── ADE_train_00000001.png
|   |   ├── ADE_train_00000002.png
|   |   ├── ...
│   ├── ade20k_panoptic_train.json
│   ├── ade20k_panoptic_val
|   |   ├── ADE_val_00000001.png
|   |   ├── ADE_val_00000002.png
|   |   ├── ...
│   ├── ade20k_panoptic_val.json
│   ├── annotations
|   |   ├── training
|   |   |   ├── ADE_train_00000001.png
|   |   |   ├── ADE_train_00000002.png
|   |   |   ├── ...
|   |   ├── validation
|   |   |   ├── ADE_val_00000001.png
|   |   |   ├── ADE_val_00000002.png
|   |   |   ├── ...
│   ├── annotations_instance
|   |   ├── training
|   |   |   ├── ADE_train_00000001.png
|   |   |   ├── ADE_train_00000002.png
|   |   |   ├── ...
|   |   ├── validation
|   |   |   ├── ADE_val_00000001.png
|   |   |   ├── ADE_val_00000002.png
|   |   |   ├── ...
│   ├── categoryMapping.txt
│   ├── images
│   |   ├── training
|   |   |   ├── ADE_train_00000001.jpg
|   |   |   ├── ADE_train_00000002.jpg
|   |   |   ├── ...
|   |   ├── validation
|   |   |   ├── ADE_val_00000001.jpg
|   |   |   ├── ADE_val_00000002.jpg
|   |   |   ├── ...
│   ├── imgCatIds.json
│   ├── objectInfo150.txt
|   |── sceneCategories.txt
```

上述文件夹包括ADE20K的语义分割、实例分割和泛在分割的所有数据。

### 从 OpenDataLab 中下载

[OpenDataLab](https://opendatalab.com/) 为人工智能研究者提供免费开源的数据集，通过 OpenDataLab，研究者可以获得格式统一的各领域经典数据集。通过平台的搜索功能，研究者可以迅速便捷地找到自己所需数据集；通过平台的统一格式，研究者可以便捷地对跨数据集任务进行开发。

目前，MIM 支持使用一条命令行从 OpenDataLab 中下载 VOC 和 COCO 数据集，后续将支持更多数据集。你也可以直接访问 OpenDataLab 平台下载你所需的数据集，然后将其转化为 MMDetection 所要求的格式。

如果使用 MIM 下载，请确保版本大于 v0.3.8，你可以使用如下命令更新:

```Bash
pip install -U openmim
```

```Bash
# install OpenDataLab CLI tools
pip install -U opendatalab
# log in OpenDataLab, registry
odl login

# download voc2007 and preprocess by MIM
mim download mmdet --dataset voc2007

# download voc2012 and preprocess by MIM
mim download mmdet --dataset voc2012

# download coco2017 and preprocess by MIM
mim download mmdet --dataset coco2017
```
