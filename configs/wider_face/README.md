# WIDER Face Dataset

[DATASET]

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
