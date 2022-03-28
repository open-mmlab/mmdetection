# Fast R-CNN

> [Fast R-CNN](https://arxiv.org/abs/1504.08083)

<!-- [ALGORITHM] -->

## Abstract

This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882189-6258c05c-f2a1-4320-9282-7e2f2d502eb2.png"/>
</div>

## Introduction

Before training the Fast R-CNN, you should first train an [RPN](../rpn/README.md), and use the RPN to extract the region proposals of the test set by this command:
```bash
./tools/dist_test.sh \
    configs/rpn_r50_fpn_1x_coco.py \
    checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth \
    8 \
    --out rpn_r50_fpn_1x_test2017.pkl
```

Then, you also need to modify the path of test set in the RPN config in order to generate `rpn_r50_fpn_1x_train2017.pkl` and `rpn_r50_fpn_1x_val2017.pkl`.
Finally, setting the path of proposals in your Fast R-CNN config. And you can start training the Fast R-CNN now.

## Results and Models

## Citation

```latex
@inproceedings{girshick2015fast,
  title={Fast r-cnn},
  author={Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2015}
}
```
