# SoftTeacher

> [End-to-End Semi-Supervised Object Detection with Soft Teacher](https://arxiv.org/abs/2106.09018)

<!-- [ALGORITHM] -->

## Abstract

This paper presents an end-to-end semi-supervised object detection approach, in contrast to previous more complex multi-stage methods. The end-to-end training gradually improves pseudo label qualities during the curriculum, and the more and more accurate pseudo labels in turn benefit object detection training. We also propose two simple yet effective techniques within this framework: a soft teacher mechanism where the classification loss of each unlabeled bounding box is weighed by the classification score produced by the teacher network; a box jittering approach to select reliable pseudo boxes for the learning of box regression. On the COCO benchmark, the proposed approach outperforms previous methods by a large margin under various labeling ratios, i.e. 1%, 5% and 10%. Moreover, our approach proves to perform also well when the amount of labeled data is relatively large. For example, it can improve a 40.9 mAP baseline detector trained using the full COCO training set by +3.6 mAP, reaching 44.5 mAP, by leveraging the 123K unlabeled images of COCO. On the state-of-the-art Swin Transformer based object detector (58.9 mAP on test-dev), it can still significantly improve the detection accuracy by +1.5 mAP, reaching 60.4 mAP, and improve the instance segmentation accuracy by +1.2 mAP, reaching 52.4 mAP. Further incorporating with the Object365 pre-trained model, the detection accuracy reaches 61.3 mAP and the instance segmentation accuracy reaches 53.0 mAP, pushing the new state-of-the-art.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/186086683-f8a69813-d09c-4c3f-a86a-e233a708cd38.png"/>
</div>

## Results and Models

|    Model    |   Detector   | Labeled Dataset | Iteration | box AP |                                  Config                                   |                                                                                                                                                                                                            Download                                                                                                                                                                                                            |
| :---------: | :----------: | :-------------: | :-------: | :----: | :-----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SoftTeacher | Faster R-CNN |     COCO-1%     |   180k    |  19.9  | [config](./soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_233412-3c8f6d4a.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_233412.log.json) |
| SoftTeacher | Faster R-CNN |     COCO-2%     |   180k    |  24.9  | [config](./soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_020244-c0d2c3aa.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_020244.log.json) |
| SoftTeacher | Faster R-CNN |     COCO-5%     |   180k    |  30.4  | [config](./soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_070656-308798ad.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230331_070656.log.json) |
| SoftTeacher | Faster R-CNN |    COCO-10%     |   180k    |  33.8  | [config](./soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py)  |  [model](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_232113-b46f78d0.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0_20230330_232113.log.json)  |

## Citation

```latex
@article{xu2021end,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
