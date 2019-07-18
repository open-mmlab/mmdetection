
# MMDetection with Robustness Benchmarking


## Introduction 

This repository contains a fork of the [mmdetection](https://github.com/open-mmlab/mmdetection) 
toolbox with code to test models on coprrupted images. It was created as a part of the 
[Robust Detection Benchmark Suite](https://github.com/bethgelab/robust_detection_benchmark) and has been 
submitted to mmdetection as [pull request](https://github.com/open-mmlab/mmdetection/pulls).

The benchmarking toolkit is part of the paper [Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484).

For more information how to evaluate models on corrupted images and results for a set of standard
models please refer to [ROBUSTNESS_BENCHMARKING.md](ROBUSTNESS_BENCHMARKING.md).

![image corruption example](demo/corruptions_sev_3.png)


## mmdetection Readme

For informations on mmdetection please refer to the [mmdetection readme](MMDETECTION_README.md).


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Robustness Benchmark

Results for standard models are available in [ROBUSTNESS_BENCHMARKING.md](ROBUSTNESS_BENCHMARKING.md).
For up-to-date results have a look at the 
[official benchmark homepage](https://github.com/bethgelab/robust_detection_benchmark).


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of MMDetection.


## Evaluating Robustness


Plase see [ROBUSTNESS_BENCHMARKING.md](ROBUSTNESS_BENCHMARKING.md) for instructions on robustness benchmarking.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{michaelis2019winter,
  title={Benchmarking Robustness in Object Detection: 
    Autonomous Driving when Winter is Coming},
  author={Michaelis, Claudio and Mitzkus, Benjamin and 
    Geirhos, Robert and Rusak, Evgenia and 
    Bringmann, Oliver and Ecker, Alexander S. and 
    Bethge, Matthias and Brendel, Wieland},
  journal={arXiv:1907.07484},
  year={2019}
}
```

## Contact

This repo is currently maintained by Claudio Michaelis ([@michaelisc](https://github.com/michaelisc)).

For questions regarding mmdetection please visit the [official repository](https://github.com/open-mmlab/mmdetection).