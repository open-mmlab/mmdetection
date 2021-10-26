# Custom object detector

Custom object detectors are object detection models with accuracy-complexity trade-off that have been pre-trained on MS COCO object detection dataset.
It is assumed that one will use these pre-trained models as starting points in order to train specific object detection models (e.g. 'cat' and 'dog' detection).
*NOTE* There was no goal to train top-scoring lightweight 80 class (MS COCO classes) detectors here,
but rather provide pre-trained checkpoints for further fine-tuning on a target dataset.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | GPU_NUM |
| --- | --- | --- | --- | --- |
| [mobilenetV2_SSD](./mobilenetV2_SSD/template.yaml) | 4.86 | 1.99 | 0.9 | 2 |
| [mobilenetV2_ATSS](./mobilenetV2_ATSS/template.yaml) | 10.43 | 2.37 | 32.5 | 1 |
| [resnet50_VFNet](./resnet50_VFNet/template.yaml) | 202.04 | 32.67 | 40.2 | 4 |
