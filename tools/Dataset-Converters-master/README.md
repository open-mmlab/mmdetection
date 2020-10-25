# Dataset Converters

Dataset Converters is a conversion toolset between different object detection and instance segmentation annotation formats.<br>
It was written and is maintained by deep learning developers from <a href="https://issivs.com/">Intelligent Security Systems</a> company to simplify the research. 

## Introduction

There are multiple different dataset annotation formats for object detection and instance segmentation. <br> 
This repository contains a system of scripts, which simplify the conversion process between those formats. We rely on COCO format as the main representation.


## Installation 

Please, ```cd``` to ```DatasetConverters``` folder and then type

```
pip install -r requirements.txt
```

This will install the required dependencies.


## Usage

<h4>Conversion</h4>

To perform conversion between different dataset formats, we provide the script called ```convert.py```.

For example, suppose, you have ADE20K dataset and you want to convert it into COCO format.<br>
For that purpose, please type

```
python convert.py --input-folder <path_to_folder_ADE20K_2016_07_26> --output-folder <output_path> \
                  --input-format ADE20K --output-format COCO --copy
```

<i>Note.</i> The shorter version of the same can be written as

```
python convert.py -i <path_to_folder_ADE20K_2016_07_26> -o <output_path> -I ADE20K -O COCO --copy
```

<i>Note.</i> <b>--copy</b> argument stands for copying images. In Linux you can instead pass <b>--symlink</b> to create symbolic links.

You are ready to use ADE20K in frameworks with COCO input format.<br>

For the full list of supported conversions, please refer to [Supported conversions](#supported-conversions) section.

<h4>Merging</h4>

If you have multiple annotations, converted to COCO format, we provide script ```merge_json_datasets.py``` to merge them.<br>
Suppose, you have COCO and Pascal VOC segmentations in COCO format and want to merge dog and horse annotations from them.
This is how ```merge_json_datasets.py``` can serve that purpose

```

python merge_json_datasets.py -d <coco_images_folder> -a <coco_annotations.json> --ids 18 19 \
                              -d <vocsegm_images_folder> -a <vocsegm_annotations.json> --ids 12 13 \
                              --output-ids 1 2 -o <output_dir> -n dog horse

```

In this example, number of merged datasets is two, but it is not limited. You can merge as many datasets and classes in COCO format, as you need.<br>
For each dataset in COCO format, one should provide the following arguments

<ul>
<li>
<b>-d</b> for images;<br>
<li>
<b>-a</b> for json file of annotations;<br>
<li>
<b>--ids</b> for list of ids of goal classes in the dataset.<br>
</ul>

After all datasets are specified with this pattern, output information is specified with the following arguments

<ul>
<li>
<b>--output-ids</b> for list of output ids of the goal classes;<br>
<li>
<b>-o</b> for output directory for the merged dataset;<br>
<li>
<b>-n</b> for names of the goal classes in the merged dataset.<br>
</ul>


## Supported conversions

In this section we list all of the supported formats and their conversions.

![](doc/dataset_converter_vis.png)

<ul>

<li>
<a href="http://groups.csail.mit.edu/vision/datasets/ADE20K/"><b>ADE20K</b></a>
<br><br>

Can be directly converted to 

<ul>
<li> COCO
</ul>

<li>

<a href="https://www.cityscapes-dataset.com/"><b>CITYSCAPES</b></a>
<br><br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

<li>

<a href="http://cocodataset.org/#home"><b>COCO</b></a>
<br><br>
Can be directly converted to 

<ul>
<li> TDG
<li> TDGSEGM
<li> VOCCALIB
</ul>

<i>Note.</i><br>
We expect names of the json annotation files correspond to names of their image folders.
If annotation file is called ```XYZ.json```, the corresponding folder is expected to be
called ```XYZ```.<br>
To convert original COCO dataset, please rename folders<br>
```train2017``` to ```instances_train2017```;<br>
```val2017``` to ```instances_val2017```<br>
and leave only two corresponding files in ```annotations``` folder: ```instances_train2017.json``` and ```instances_val2017.json```.

<li>
<a href="https://github.com/opencv/cvat"><b>CVAT</b></a>
<br><br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

<i>Note.</i><br>
In case of CVAT input format, we expect the xml annotation file and 
images to be placed in the same folder. That folder is supposed to be 
```input_folder``` argument of ```convert``` function.

<li>
<a href="https://storage.googleapis.com/openimages/web/index.html"><b>OID</b></a>
<br><br>
Stands for Open Images Dataset V4.<br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

<li>
<b>TDG</b>
<br><br>
Custom format for bounding box annotation.<br>
You can use it to train Faster R-CNNs and SSDs in their Caffe branches.<br>
Can be directly converted to 

<ul>
<li> COCO
<li> <a href="https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev">FRCNN</a>
<li> <a href="https://github.com/weiliu89/caffe/tree/ssd">SSD</a>
</ul>

<li>
<b>TDGSEGM</b>
<br><br>
Custom format for instance segmnentation.<br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

<li>
<a href="http://host.robots.ox.ac.uk/pascal/VOC/"><b>VOC</b></a>
<br><br>
Stands for bounding box annotations from Pascal VOC datasets.<br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

<li>
<b>VOCCALIB</b>
<br><br>
Stands for bounding box annotations used in <a href="https://software.intel.com/en-us/openvino-toolkit">OpenVINO</a> calibration tool.<br>
They are supposed to be "VOC-like". Convert to this format to use the result in OpenVINO 
calibration tool.<br>
No conversions from this format available.


<li>
<a href="http://host.robots.ox.ac.uk/pascal/VOC/"><b>VOCSEGM</b></a>
<br><br>
Stands for instance segmentation annotations from Pascal VOC datasets.<br>
Can be directly converted to 

<ul>
<li> COCO
</ul>

</ul>

# How to contribute

We welcome community contributions to the Dataset Converters.<br>

If you want to add a new dataset converter, please note, that we expect

<ul>
<li> The new dataset is free and open, so we are able to download it and test your code;
<li> Your code is written from scratch and does not have parts copied from other repositories.
</ul>

The list of the core files, which are the key to understand the implementation process is the following

```
Converter.py
ConverterBase.py
converters.py
formats.py
```

The new converter is a subclass of ```ConverterBase``` class with ```_run``` mehtod
overloaded and conversion format added to the list ```formats```.



