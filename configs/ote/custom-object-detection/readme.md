# Custom object detector

Custom object detectors are lightweight object detection models that have been pre-trained on MS COCO object detection dataset.
It is assumed that one will use these pre-trained models as starting points in order to train specific object detection models (e.g. 'cat' and 'dog' detection).
*NOTE* There was no goal to train top-scoring lightweight 80 class (MS COCO classes) detectors here,
but rather provide pre-trained checkpoints for further fine-tuning on a target dataset.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v2-2s_ssd-256x256 | 0.86 | 1.99 | 11.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-256x256.pth), [config](./mobilenet_v2-2s_ssd-256x256/template.yaml) | 3 |
| mobilenet_v2-2s_ssd-384x384 | 1.92 | 1.99 | 13.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-384x384.pth), [config](./mobilenet_v2-2s_ssd-384x384/template.yaml) | 3 |
| mobilenet_v2-2s_ssd-512x512 | 3.42 | 1.99 | 12.7 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-512x512.pth), [config](./mobilenet_v2-2s_ssd-512x512/template.yaml) | 3 |
