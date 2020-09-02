# VarifocalNet: An IoU-aware Dense Object Detector

## Introduction
**VarifocalNet (VFNet)** learns to predict the IoU-aware classification score which mixes the object presence confidence and localization accuracy together as the detection score for a bounding box. The learning is supervised by the proposed Varifocal Loss (VFL), based on a new star-shaped bounding box feature representation (the features at nine yellow sampling points). Given the new representation, the object localization accuracy is further improved by refining the initially regressed bounding box. The full paper is available at: [https://arxiv.org/abs/2008.13367](https://arxiv.org/abs/2008.13367).

<div align="center">
  <img src="VFNet.png" width="600px" />
  <p>Learning to Predict the IoU-aware Classification Score.</p>
</div>

## Citing VarifocalNet

```
@article{zhang2020varifocalnet,
  title={VarifocalNet: An IoU-aware Dense Object Detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and S{\"u}nderhauf, Niko},
  journal={arXiv preprint arXiv:2008.13367},
  year={2020}
}
```

## Results and Models

| Backbone     | Style     | DCN     | MS train | Lr schd |Inf time (fps) | box AP (val) | box AP (test-dev) | Download |
|:------------:|:---------:|:-------:|:--------:|:-------:|:-------------:|:------------:|:-----------------:|:--------:|
| R-50         | pytorch   | N       | N        | 1x      | 19.4          | 41.6         | 41.6              | [model](https://drive.google.com/file/d/1aF3Fi5rYeMqSC3Ndo4VEqjPXk4fcOfjt/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/16tufMtxWI5Zq2Bx9VNt57Fxt8VnlBmnR/view?usp=sharing)|
| R-50         | pytorch   | N       | Y        | 2x      | 19.3          | 44.5         | 44.8              | [model](https://drive.google.com/file/d/1oAUu7zGZmPni0XZu8XJMDWR5GJi_pwYz/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1sFlRVMyTGcZBmnMxx1T_ulNHTC_JoynF/view?usp=sharing)|
| R-50         | pytorch   | Y       | Y        | 2x      | 16.3          | 47.8         | 48.0              | [model](https://drive.google.com/file/d/16rk1pCPmQOj98GkpM-y3IPBpdKcfq0I4/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pcOW7CjJYz0XHRkZKi_TjoD4hj4mz6DP/view?usp=sharing)|
| R-101        | pytorch   | N       | N        | 1x      | 15.5          | 43.0         | 43.6              | [model](https://drive.google.com/file/d/1z76RBD6fI43IQn2tGuSJ5KonuJE49Nfk/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ASitqJL2puajfpFVQv04VVKWSvkWqIKq/view?usp=sharing)|
| R-101        | pytorch   | N       | N        | 2x      | 15.6          | 43.5         | 43.9              | [model](https://drive.google.com/file/d/1SXYCfkOXXGBhvURcqH23DA5gIwSFYz2u/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1abueLX2H4R-zr5kMw1WDbzf4wSQ2my3_/view?usp=sharing)|
| R-101        | pytorch   | N       | Y        | 2x      | 15.6          | 46.2         | 46.7              | [model](https://drive.google.com/file/d/1ioQ2Fdbp4OS2Oi6g7no6fqU2-V8TtOTz/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1XA9_xH0TmOJgsbAQ05MV1AdcS6AkKvfz/view?usp=sharing)|
| R-101        | pytorch   | Y       | Y        | 2x      | 12.6          | 49.0         | 49.2              | [model](https://drive.google.com/file/d/1W-Wkl3e3f64PzP8vJ0iB5svMqNlOzPyI/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1qvKopNIXZsexEMBXP7TA_agxw-qul6id/view?usp=sharing)|
| X-101-32x4d  | pytorch   | N       | Y        | 2x      | 13.1          | 47.4         | 47.6              | [model](https://drive.google.com/file/d/1X-soI6dyFxv0jWOKxthPPsVeyqIRRa-F/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ywK17-fJYWqTabl8NE-JOFKhk6ihXy25/view?usp=sharing)|
| X-101-32x4d  | pytorch   | Y       | Y        | 2x      | 10.1          | 49.7         | 50.0              | [model](https://drive.google.com/file/d/1QtMyI4tjccigDPn2A-V1sQuK-Z6eTHwq/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1OqEIS4wnBdkSLTVn2mLhyOEef0yuNZyg/view?usp=sharing)|
| X-101-64x4d  | pytorch   | N       | Y        | 2x      |  9.2          | 48.2         | 48.5              | [model](https://drive.google.com/file/d/1m0BQ6XyAlxNdJQVbg_4OuEgEctywPVx9/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1DFgBSzWyIaPo7UZsY2oRhxq1-W0gre8Z/view?usp=sharing)|
| X-101-64x4d  | pytorch   | Y       | Y        | 2x      |  6.7          | 50.4         | 50.8              | [model](https://drive.google.com/file/d/1GkyJirn8J10TTyWiyw5C4boKWlW9epSU/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ZPPiX1KhT6D48OPSOnPPGa9NqPa4HePG/view?usp=sharing)|
| R2-101       | pytorch   | N       | Y        | 2x      | 13.0          | 49.2         | 49.3              | [model](https://drive.google.com/file/d/1E4o1CxaWUQV7-HAyqbITw7JD8mOF7tNW/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ESnWn7nXRJVcqQb5OjH3c6XM8Rqc4shI/view?usp=sharing)|
| R2-101       | pytorch   | Y       | Y        | 2x      | 10.3          | 51.1         | 51.3              | [model](https://drive.google.com/file/d/1kCiEqAA_VQlhbiNuZ3HWGhBD1JvVpK0c/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1BTwm-knCIT-kzkASjWNMfRWaAwI0ONmC/view?usp=sharing)|


**Notes:**
- The MS-train scale range is 1333x[480:960] (`range` mode) and the inference scale keeps 1333x800.
- The R2-101 backbone is Res2Net-101. If you use it with DCN in inference, please comment [`delattr(self, 'conv2')`](https://github.com/open-mmlab/mmdetection/blob/6acf6be921429343f885c0fab7fa75ba01c46d16/mmdet/models/backbones/res2net.py#L101) in res2net.py.
- DCN means using `DCNv2` in both backbone and head.
- Inference time is tested with a Nvidia V100 GPU on HPC ([log file](https://drive.google.com/file/d/1dc9296G6JevouLixj-g81VgccEt54ceP/view?usp=sharing)).
- More results and pre-trained models can be found in [VarifocalNet-Github](https://github.com/hyz-xmaster/VarifocalNet)
