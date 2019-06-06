## WIDER Face Dataset

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

After that you can train the SSD300 on WIDER Face by launching training with the `ssd300_wider_face.py` or `mobilenetv2_tiny_ssd300_wider_face.py` configs or create your own config based on the presented one.

## Baseline Models

To download pre-trained MobileNetV2 backbone visit this [page](https://github.com/tonylins/pytorch-mobilenet-v2).

| Model  | Width factor  | Complexity (GMACS) | Parameters (M) | AP* (all faces) | AP* (faces > 30 pix)| AP* (faces > 60 pix) | AP* (faces > 100 pix)| Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|
| MobileNetV2-SSD-Lite-SingleHead  | 0.75 |   0.51    | 1.03      | 0.305 | 0.768 | 0.867 | 0.927 | [model](https://drive.google.com/file/d/1jRi0hxxzIlfgEEoKHS_SkOD529y3kKNb/view?usp=sharing) |
| MobileNetV2-SSD-Lite-SingleHead  | 1.0  |   0.87    | 1.79      | 0.319 | 0.790 | 0.885 | 0.935 | [model](https://drive.google.com/file/d/1jjQBkLsNxfNR29UsJ2aeq1Ir4oZTuKEl/view?usp=sharing) |
| VGG16-SSD  | 1.0  |   30.49    | 23.75      | 0.337 | 0.841 | 0.920 | 0.941 | [model](https://drive.google.com/file/d/1tBSPmUQwJZTAVfuxRlvDNkJVm2ww1gMk/view?usp=sharing) |
*VOC AP on WIDER val with filtered faces by size threshold
