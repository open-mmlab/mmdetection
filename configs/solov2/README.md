# SOLOv2

> [SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

<!-- [ALGORITHM] -->

## Abstract

In this work, we aim at building a simple, direct, and fast instance segmentation
framework with strong performance. We follow the principle of the SOLO method of
Wang et al. "SOLO: segmenting objects by locations". Importantly, we take one
step further by dynamically learning the mask head of the object segmenter such
that the mask head is conditioned on the location. Specifically, the mask branch
is decoupled into a mask kernel branch and mask feature branch, which are
responsible for learning the convolution kernel and the convolved features
respectively. Moreover, we propose Matrix NMS (non maximum suppression) to
significantly reduce the inference time overhead due to NMS of masks. Our
Matrix NMS performs NMS with parallel matrix operations in one shot, and
yields better results. We demonstrate a simple direct instance segmentation
system, outperforming a few state-of-the-art methods in both speed and accuracy.
A light-weight version of SOLOv2 executes at 31.3 FPS and yields 37.1% AP.
Moreover, our state-of-the-art results in object detection (from our mask byproduct)
 and panoptic segmentation show the potential to serve as a new strong baseline
 for many instance-level recognition tasks besides instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/167235090-f20dab74-43a5-44ed-9f11-4e5f08866f45.png"/>
</div>

## Results and Models

### SOLOv2

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP| Config   |Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|:--------:|
| R-50      | pytorch | N        | 1x      |          |                |        |          |[model]() &#124; [log]() |
| R-50      | pytorch | Y        | 3x      |          |                |        |          |[model]() &#124; [log]() |
| R-101     | pytorch | Y        | 3x      |          |                |        |          |[model]() &#124; [log]() |
| R-101(DCN)| pytorch | Y        | 3x      |          |                |        |          |[model]() &#124; [log]() |
| X-101(DCN)| pytorch | Y        | 3x      |          |                |        |          |[model]() &#124; [log]() |

### Light SOLOv2

| Backbone       | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Config   | Download |
|:--------------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:-------:|:--------:|:--------:|
| R-18           | pytorch |     Y    | 3x      |          |                |         |          |  [model]() &#124; [log]() |
| R-34           | pytorch |     Y    | 3x      |          |                |         |          |  [model]() &#124; [log]() |
| R-50           | pytorch |     Y    | 3x      |          |                |         |          |  [model]() &#124; [log]() |
| R-50(DCN)      | pytorch |     Y    | 3x      |          |                |         |          |  [model]() &#124; [log]() |


## Citation

```latex
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
