# AutoAssign: Differentiable Label Assignment for Dense Object Detection

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50     | pytorch | 1x      | 4.08      |   40.4  |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.3 mAP. mAP 40.3 ~ 40.6 is acceptable.
2. You can get a more stable results ~ mAP 40.6 with a schedule total 13 epoch, and learning rate is divided by 10 at 10 and 13 epoch.
