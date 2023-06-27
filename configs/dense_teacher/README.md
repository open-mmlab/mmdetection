# DenseTeacher

> [Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection](https://arxiv.org/abs/2207.02541v2)

<!-- [ALGORITHM] -->

## Abstract

To date, the most powerful semi-supervised object detectors (SS-OD) are based on pseudo-boxes, which need a sequence of
post-processing with fine-tuned hyper-parameters. In this work, we propose replacing the sparse pseudo-boxes with the dense prediction as
a united and straightforward form of pseudo-label. Compared to the pseudo-boxes, our Dense Pseudo-Label (DPL) does not involve any post-
processing method, thus retaining richer information. We also introduce a region selection technique to highlight the key information while sup-
pressing the noise carried by dense labels. We name our proposed SSOD algorithm that leverages the DPL as Dense Teacher. On COCO
and VOC, Dense Teacher shows superior performance under various settings compared with the pseudo-box-based methods. Code is available
at https://github.com/Megvii-BaseDetection/DenseTeacher

<div align=center>
<img src=""/>
</div>

## Citation

```latex
@inproceedings{zhou2022dense,
  title={Dense teacher: Dense pseudo-labels for semi-supervised object detection},
  author={Zhou, Hongyu and Ge, Zheng and Liu, Songtao and Mao, Weixin and Li, Zeming and Yu, Haiyan and Sun, Jian},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IX},
  pages={35--50},
  year={2022},
  organization={Springer}
}
```
