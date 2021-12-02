# WIDER FACE: A Face Detection Benchmark

## Abstract

<!-- [ABSTRACT] -->

Face detection is one of the most studied topics in the computer vision community. Much of the progresses have been made by the availability of face detection benchmark datasets. We show that there is a gap between current face detection performance and the real world requirements. To facilitate future face detection research, we introduce the WIDER FACE dataset, which is 10 times larger than existing datasets. The dataset contains rich annotations, including occlusions, poses, event categories, and face bounding boxes. Faces in the proposed dataset are extremely challenging due to large variations in scale, pose and occlusion, as shown in Fig. 1. Furthermore, we show that WIDER FACE dataset is an effective training source for face detection. We benchmark several representative detection systems, providing an overview of state-of-the-art performance and propose a solution to deal with large scale variation. Finally, we discuss common failure cases that worth to be further investigated.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144000364-3320de79-34fc-40a6-938f-bb512f05a4bb.png" height="400"/>
</div>

<!-- [PAPER_TITLE: WIDER FACE: A Face Detection Benchmark] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1511.06523] -->

## Introduction

<!-- [DATASET] -->

To use the WIDER Face dataset you need to download it
and extract to the `data/WIDERFace` folder. Annotation in the VOC format
can be found in this [repo](https://github.com/sovrasov/wider-face-pascal-voc-annotations.git).
You should move the annotation files from `WIDER_train_annotations` and `WIDER_val_annotations` folders
to the `Annotation` folders inside the corresponding directories `WIDER_train` and `WIDER_val`.
Also annotation lists `val.txt` and `train.txt` should be copied to `data/WIDERFace` from `WIDER_train_annotations` and `WIDER_val_annotations`.
The directory should be like this:

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── WIDERFace
│   │   ├── WIDER_train
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── WIDER_val
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── val.txt
│   │   ├── train.txt

```

After that you can train the SSD300 on WIDER by launching training with the `ssd300_wider_face.py` config or
create your own config based on the presented one.

## Citation

```
@inproceedings{yang2016wider,
   Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
   Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   Title = {WIDER FACE: A Face Detection Benchmark},
   Year = {2016}
}
```
