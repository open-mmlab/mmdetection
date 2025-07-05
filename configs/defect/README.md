# Industrial Surface Defect Detection (NEU)

This folder contains a complete example of training and testing a surface defect detection model on the [NEU Surface Defect Database](https://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html).

---

## What’s included

- **`faster-rcnn_r50_fpn_1x_neu.py`** — Config file that adapts MMDetection’s standard Faster R-CNN to detect 6 types of surface defects (e.g., scratches, inclusions).
- **`convert_annotations.py`** — Utility script to help convert the raw NEU images into COCO-style annotations (train, val, test splits).
- (Optional) **`app.py`** — Example Streamlit app to test your trained model interactively.

---

## Why this is useful

Many factories still rely on manual visual inspection for surface quality. This simple example shows how to adapt MMDetection to a small real-world manufacturing dataset and helps engineers and students see how to structure their own quality inspection projects.

---

## How to use

1. Download the NEU dataset and extract it under `data/NEU/`.

2. Use `convert_annotations.py` to split and convert your images:
