# Probabilistic Anchor Assignment with IoU Prediction for Object Detection



## Results and Models
We provide config files to reproduce the object detection results in the
ECCV 2020 paper for Probabilistic Anchor Assignment with IoU
Prediction for Object Detection.

| Backbone    | Lr schd | Mem (GB) | Score voting | box AP | Download |
|:-----------:|:-------:|:--------:|:------------:|:------:|:--------:|
| R-50-FPN    | 12e     | 3.7     | True          | 40.4   | [model]() &#124; [log]() |
| R-50-FPN    | 12e     | 3.7     | False         | 40.2   | - |
| R-50-FPN    | 24e     | 3.7     | True          | 41.6   | [model]() &#124; [log]() |
| R-101-FPN   | 12e     | 6.2     | True          | 42.6   | [model]() &#124; [log]() |
| R-101-FPN   | 12e     | 6.2     | False         | 42.4   | - |
| R-101-FPN   | 24e     | 6.2     | True          | 43.5   | [model]() &#124; [log]() |

**Note**:
1. Compared with origin results in the paper, the drop is mainly due to a [issue](https://github.com/kkhoot/PAA/issues/8).
2. We find that the performance is unstable with 1x setting and may fluctuate by about 0.2 mAP. We report the best results.
