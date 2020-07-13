# YOLOv3

## Introduction
```
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and Models

|    Backbone     | Train Scale  | Lr schd | Mem (GB) | Eval Scale | Inf time (fps) | box AP | Download |
| :-------------: | :----------: | :-----: | :------: | :--------: | :------------: | :----: |:--------:|
|   DarkNet-53    | Multi-Scale  |  273e   |   1.8    | 608 * 608  |      44        |**37.6**| [model](https://drive.google.com/file/d/1Ca27fP4hlBFduMCv5b_f-0J9EdfxCgPb/view?usp=sharing) &#124; [log](https://github.com/open-mmlab/mmdetection/files/4910982/log.zip) |
|        -        |      -       |     -   |  -       | 416 * 416  |      **64**    |  34.8  | - |


## Credit
This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.
