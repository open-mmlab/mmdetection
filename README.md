## Requirements
Tested on the following environments
- mmdetection==2.26.0
- torch==1.11.0
- cuda 11.3


## Structures
- Original FasterRCNN
  - detectors: "Faster_RCNN" ($mmdet/models/detectors/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "StandardRoIHead" ($mmdet/models/roi_heads/standard_roi_head.py)
  - data: "CocoDataset" ($mmdet/datasets/coco.py)

- MSDET FasterRCNN
  - detectors: "Faster_RCNN_TS" ($msdet/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "ContRoIHead" ($msdet/roi_heads.py)
  - data: "CocoConDataset" ($msdet/coco.py)


## Train and Evaluation
- Train Code
  ```
  source tools/dist_train.sh
  ```