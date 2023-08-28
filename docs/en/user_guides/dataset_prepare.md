# Dataset Prepare

### Basic Detection Dataset Preparation

MMDetection supports multiple public datasets including COCO, Pascal VOC, CityScapes, and [more](../../../configs/_base_/datasets).

Public datasets like [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) or mirror and [COCO](https://cocodataset.org/#download) are available from official websites or mirrors. Note: In the detection task, Pascal VOC 2012 is an extension of Pascal VOC 2007 without overlap, and we usually use them together.
It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to `$MMDETECTION/data` as below.
If your folder structure is different, you may need to change the corresponding paths in config files.

We provide a script to download datasets such as COCO, you can run `python tools/misc/download_dataset.py --dataset-name coco2017` to download COCO dataset.
For users in China, more datasets can be downloaded from the opensource dataset platform: [OpenDataLab](https://opendatalab.com/?source=OpenMMLab%20GitHub).

For more usage please refer to [dataset-download](./useful_tools.md#dataset-download)

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

Some models require additional [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) datasets, such as HTC, DetectoRS and SCNet, you can download, unzip, and then move them to the coco folder. The directory should be like this.

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

Panoptic segmentation models like PanopticFPN require additional [COCO Panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) datasets, you can download, unzip, and then move them to the coco annotation folder. The directory should be like this.

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

The [cityscapes](https://www.cityscapes-dataset.com/) annotations need to be converted into the coco format using `tools/dataset_converters/cityscapes.py`:

```shell
pip install cityscapesscripts

python tools/dataset_converters/cityscapes.py \
    ./data/cityscapes \
    --nproc 8 \
    --out-dir ./data/cityscapes/annotations
```

### COCO Caption Dataset Preparation

COCO Caption uses the COCO2014 dataset image and uses the annotation of karpathy.

At first, you need to download the COCO2014 dataset.

```shell
python tools/misc/download_dataset.py --dataset-name coco2014 --unzip
```

The dataset will be downloaded to `data/coco` under the current path. Then download the annotation of karpathy.

```shell
cd data/coco/annotations
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json
```

The final directory structure of the dataset folder that can be directly used for training and testing is as follows:

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

### COCO Semantic Dataset Preparation

There are two types of annotations for COCO semantic segmentation, which differ mainly in the definition of category names, so there are two ways to handle them. The first is to directly use the stuffthingmaps dataset, and the second is to use the panoptic dataset.

**(1) Use stuffthingmaps dataset**

The download link for this dataset is [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip). Please download and extract it to the `data/coco` folder.

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

This dataset is different from the standard COCO category annotation in that it includes 172 classes: 80 "thing" classes, 91 "stuff" classes, and 1 "unlabeled" class. The description of each class can be found at https://github.com/nightrome/cocostuff/blob/master/labels.md.

Although only 172 categories are annotated, the maximum label ID in `stuffthingmaps` is 182, and some categories in the middle are not annotated. In addition, the "unlabeled" category of class 0 is removed. Therefore, the relationship between the value at each position in the final `stuffthingmaps` image can be found at https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/cocostuff/labels.txt.

To train efficiently and conveniently for users, we need to remove 12 unannotated classes before starting training or evaluation. The names of these 12 classes are: `street sign, hat, shoe, eye glasses, plate, mirror, window, desk, door, blender, hair brush`. The category information that can be used for training and evaluation can be found in `mmdet/datasets/coco_semantic.py`.

You can use `tools/dataset_converters/coco_stuff164k.py` to convert the downloaded `stuffthingmaps` to a dataset that can be directly used for training and evaluation. The directory structure of the converted dataset is as follows:

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

`stuffthingmaps_semseg` is the newly generated COCO semantic segmentation dataset that can be directly used for training and testing.

**(2) use panoptic dataset**

The number of categories in the semantic segmentation dataset generated through panoptic annotation will be less than that generated using the `stuffthingmaps` dataset. First, you need to prepare the panoptic segmentation annotations, and then use the following script to complete the conversion.

```shell
python tools/dataset_converters/prepare_coco_semantic_annos_from_panoptic_annos.py data/coco
```

The directory structure of the converted dataset is as follows:

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

`panoptic_semseg_train2017` and `panoptic_semseg_val2017` are the newly generated COCO semantic segmentation datasets that can be directly used for training and testing. Note that their category information is the same as that of COCO panoptic segmentation, including both "thing" and "stuff" categories.

### RefCOCO Dataset Preparation

The images and annotations of [RefCOCO](https://github.com/lichengunc/refer) series datasets can be download by running `tools/misc/download_dataset.py`:

```shell
python tools/misc/download_dataset.py --dataset-name refcoco --save-dir data/coco --unzip
```

Then the directory should be like this:

```text
data
├── coco
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   ├── refcocog
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(umd).p
│   │── train2014
```

### ADE20K 2016 Dataset Preparation

The images and annotations of [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset can be download by running `tools/misc/download_dataset.py`:

```shell
python tools/misc/download_dataset.py --dataset-name ade20k_2016 --save-dir data --unzip
```

Then move the annotations to the `data/ADEChallengeData2016` directory and run the preprocess script to produce the coco format annotations:

```shell
mv data/annotations_instance data/ADEChallengeData2016/
mv data/categoryMapping.txt data/ADEChallengeData2016/
mv data/imgCatIds.json data/ADEChallengeData2016/
python tools/dataset_converters/ade20k2coco.py data/ADEChallengeData2016 --task panoptic
python tools/dataset_converters/ade20k2coco.py data/ADEChallengeData2016 --task instance
```

The directory should be like this.

```text
data
├── ADEChallengeData2016
│   ├── ade20k_instance_train.json
│   ├── ade20k_instance_val.json
│   ├── ade20k_panoptic_train
│   │   ├── ADE_train_00000001.png
│   │   ├── ADE_train_00000002.png
│   │   ├── ...
│   ├── ade20k_panoptic_train.json
│   ├── ade20k_panoptic_val
│   │   ├── ADE_val_00000001.png
│   │   ├── ADE_val_00000002.png
│   │   ├── ...
│   ├── ade20k_panoptic_val.json
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   │   │   ├── ...
│   │   ├── validation
│   │   │   ├── ADE_val_00000001.png
│   │   │   ├── ADE_val_00000002.png
│   │   │   ├── ...
│   ├── annotations_instance
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   │   │   ├── ...
│   │   ├── validation
│   │   │   ├── ADE_val_00000001.png
│   │   │   ├── ADE_val_00000002.png
│   │   │   ├── ...
│   ├── categoryMapping.txt
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
│   │   │   ├── ...
│   │   ├── validation
│   │   │   ├── ADE_val_00000001.jpg
│   │   │   ├── ADE_val_00000002.jpg
│   │   │   ├── ...
│   ├── imgCatIds.json
│   ├── objectInfo150.txt
│   │── sceneCategories.txt
```

The above folders include all data of ADE20K's semantic segmentation, instance segmentation, and panoptic segmentation.

### Download from OpenDataLab

By using [OpenDataLab](https://opendatalab.com/), researchers can obtain free formatted datasets in various fields. Through the search function of the platform, researchers may address the dataset they look for quickly and easily. Using the formatted datasets from the platform, researchers can efficiently conduct tasks across datasets.

Currently, MIM supports downloading VOC and COCO datasets from OpenDataLab with one command line. More datasets will be supported in the future. You can also directly download the datasets you need from the OpenDataLab platform and then convert them to the format required by MMDetection.

If you use MIM to download, make sure that the version is greater than v0.3.8. You can use the following command to update:

```Bash
pip install -U openmim
```

```Bash
# install OpenXLab CLI tools
pip install -U openxlab
# log in OpenXLab, registry
openxlab login

# download voc2007 and preprocess by MIM
mim download mmdet --dataset voc2007

# download voc2012 and preprocess by MIM
mim download mmdet --dataset voc2012

# download coco2017 and preprocess by MIM
mim download mmdet --dataset coco2017
```
