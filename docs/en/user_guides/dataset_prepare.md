# Dataset Prepare

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
