# Fast R-CNN

## Abstract

<!-- [ABSTRACT] -->

This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882189-6258c05c-f2a1-4320-9282-7e2f2d502eb2.png"/>
</div>

<!-- [PAPER_TITLE: Fast R-CNN] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1504.08083] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{girshick2015fast,
  title={Fast r-cnn},
  author={Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2015}
}
```

## Results and models
