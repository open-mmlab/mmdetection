# VISION-Datasets

> VISION Datasets: A Benchmark for Vision-based InduStrial InspectiON

## Introduction

Despite progress in vision-based inspection algorithms, real-world industrial challenges – specifically in data availability, quality, and complex production requirements – often remain under-addressed. We introduce the VISION Datasets, a diverse collection of 14 industrial inspection datasets, uniquely poised to meet these challenges. Unlike previous datasets, VISION brings versatility to defect detection, offering annotation masks across all splits and catering to various detection methodologies. Our datasets also feature instance-segmentation annotation, enabling precise defect identification. With a total of 18k images encompassing 44 defect types, VISION strives to mirror a wide range of real-world production scenarios. By supporting two ongoing challenge competitions on the VISION Datasets, we hope to foster further advancements in vision-based industrial inspection. The datasets are available at https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/1d8e81d1-0023-49ce-a855-a09149509fe7" width="70%"/>
</div>

## Dataset Preparation

At first, you should download the dataset from https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets and organize it as follows:

```text
mmdetection
├── mmdet
├── tools
├── configs
├── data
├── │── VISION-Datasets
├── │   ├── Cable.tar.gz
├── │   ├── Capacitor.tar.gz
├── │   ├── Casting.tar.gz
├── |   ├── Console.tar.gz
├── │   ├── Cylinder.tar.gz
├── │   ├── Electronics.tar.gz
├── │   ├── Groove.tar.gz
├── │   ├── Hemisphere.tar.gz
├── │   ├── Lens.tar.gz
├── │   ├── PCB_1.tar.gz
├── │   ├── PCB_2.tar.gz
├── |   ├── README.md
├── │   ├── Ring.tar.gz
├── │   ├── Screw.tar.gz
├── │   └── Wood.tar.gz
```

Then you can use the following command to save the following command as the `vision_unzip.sh` file and place it in the `mmdetection` root directory, and then run the script `bash vision_unzip.sh` to unzip it.

```shell
#!/usr/bin/env bash

for file in data/VISION-Datasets/*.tar.gz; do
    tar -xzvzf "$file" -C data/VISION-Datasets/
done
```

Finally, the file organization format is as follows:

```text
mmdetection
├── mmdet
├── tools
├── configs
├── data
|   │── VISION-Datasets
|   │   ├── Cable.tar.gz
|   │   ├── Capacitor.tar.gz
|   │   ├── Casting.tar.gz
|   |   ├── Console.tar.gz
|   │   ├── Cylinder.tar.gz
|   │   ├── Electronics.tar.gz
|   │   ├── Groove.tar.gz
|   │   ├── Hemisphere.tar.gz
|   │   ├── Lens.tar.gz
|   │   ├── PCB_1.tar.gz
|   │   ├── PCB_2.tar.gz
|   |   ├── README.md
|   │   ├── Ring.tar.gz
|   │   ├── Screw.tar.gz
|   │   └── Wood.tar.gz
|   │   ├── Cable
|   │   |    |── train
|   │   |    |     |──  _annotations.coco.json # COCO format annotation
|   │   |    |     |──  000001.png # Images
|   │   |    |     |──  000002.png
|   │   |    |     |──  xxxxxx.png
|   │   |    |── val
|   │   |    |     |──  _annotations.coco.json # COCO format annotation
|   │   |    |     |──  xxxxxx.png # Images
|   │   |    |── inference
|   │   |    |     |──  _annotations.coco.json # COCO format annotation with unlabeled image list only
|   │   |    |     |──  xxxxxx.png # Images
...
```

## Models and Results

TODO

## Citation

```latex
@article{vision-datasets,
  title         = {VISION Datasets: A Benchmark for Vision-based InduStrial InspectiON},
  author        = {Haoping Bai, Shancong Mou, Tatiana Likhomanenko, Ramazan Gokberk Cinbis, Oncel Tuzel, Ping Huang, Jiulong Shan, Jianjun Shi, Meng Cao},
  journal       = {arXiv preprint arXiv:2306.07890},
  year          = {2023},
}
```
