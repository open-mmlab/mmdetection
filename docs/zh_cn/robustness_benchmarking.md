# 检测器鲁棒性检查

## 介绍

我们提供了在 [Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484) 中定义的「图像损坏基准测试」上测试目标检测和实例分割模型的工具。
此页面提供了如何使用该基准测试的基本教程。

```latex
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

![image corruption example](../resources/corruptions_sev_3.png)

## 关于基准测试

要将结果提交到基准测试，请访问[基准测试主页](https://github.com/bethgelab/robust-detection-benchmark)

基准测试是仿照 [imagenet-c 基准测试](https://github.com/hendrycks/robustness)，由 Dan Hendrycks 和 Thomas Dietterich 在[Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)(ICLR 2019)中发表。

图像损坏变换功能包含在此库中，但可以使用以下方法单独安装：

```shell
pip install imagecorruptions
```

与 imagenet-c 相比，我们必须进行一些更改以处理任意大小的图像和灰度图像。
我们还修改了“运动模糊”和“雪”损坏，以解除对于 linux 特定库的依赖，
否则必须单独安装这些库。有关详细信息，请参阅 [imagecorruptions](https://github.com/bethgelab/imagecorruptions)。

## 使用预训练模型进行推理

我们提供了一个测试脚本来评估模型在基准测试中提供的各种损坏变换组合下的性能。

### 在数据集上测试

- [x] 单张 GPU 测试
- [ ] 多张 GPU 测试
- [ ] 可视化检测结果

您可以使用以下命令在基准测试中使用 15 种损坏变换来测试模型性能。

```shell
# single-gpu testing
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

也可以选择其它不同类型的损坏变换。

```shell
# noise
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --corruptions noise

# blur
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --corruptions blur

# wetaher
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --corruptions weather

# digital
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --corruptions digital
```

或者使用一组自定义的损坏变换，例如：

```shell
# gaussian noise, zoom blur and snow
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --corruptions gaussian_noise zoom_blur snow
```

最后，我们也可以选择施加在图像上的损坏变换的严重程度。
严重程度从 1 到 5 逐级增强，0 表示不对图像施加损坏变换，即原始图像数据。

```shell
# severity 1
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --severities 1

# severities 0,2,4
python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --severities 0 2 4
```

## 模型测试结果

下表是各模型在 COCO 2017val 上的测试结果。

|        Model        |      Backbone       |  Style  | Lr schd | box AP clean | box AP corr. | box % | mask AP clean | mask AP corr. | mask % |
| :-----------------: | :-----------------: | :-----: | :-----: | :----------: | :----------: | :---: | :-----------: | :-----------: | :----: |
|    Faster R-CNN     |      R-50-FPN       | pytorch |   1x    |     36.3     |     18.2     | 50.2  |       -       |       -       |   -    |
|    Faster R-CNN     |      R-101-FPN      | pytorch |   1x    |     38.5     |     20.9     | 54.2  |       -       |       -       |   -    |
|    Faster R-CNN     |   X-101-32x4d-FPN   | pytorch |   1x    |     40.1     |     22.3     | 55.5  |       -       |       -       |   -    |
|    Faster R-CNN     |   X-101-64x4d-FPN   | pytorch |   1x    |     41.3     |     23.4     | 56.6  |       -       |       -       |   -    |
|    Faster R-CNN     |    R-50-FPN-DCN     | pytorch |   1x    |     40.0     |     22.4     | 56.1  |       -       |       -       |   -    |
|    Faster R-CNN     | X-101-32x4d-FPN-DCN | pytorch |   1x    |     43.4     |     26.7     | 61.6  |       -       |       -       |   -    |
|     Mask R-CNN      |      R-50-FPN       | pytorch |   1x    |     37.3     |     18.7     | 50.1  |     34.2      |     16.8      |  49.1  |
|     Mask R-CNN      |    R-50-FPN-DCN     | pytorch |   1x    |     41.1     |     23.3     | 56.7  |     37.2      |     20.7      |  55.7  |
|    Cascade R-CNN    |      R-50-FPN       | pytorch |   1x    |     40.4     |     20.1     | 49.7  |       -       |       -       |   -    |
| Cascade Mask R-CNN  |      R-50-FPN       | pytorch |   1x    |     41.2     |     20.7     | 50.2  |     35.7      |     17.6      |  49.3  |
|      RetinaNet      |      R-50-FPN       | pytorch |   1x    |     35.6     |     17.8     | 50.1  |       -       |       -       |   -    |
| Hybrid Task Cascade | X-101-64x4d-FPN-DCN | pytorch |   1x    |     50.6     |     32.7     | 64.7  |     43.8      |     28.1      |  64.0  |

由于对图像的损坏变换存在随机性，测试结果可能略有不同。
