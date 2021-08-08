# 模型库

## 镜像地址

从MMDetection V2.0起，我们只通过阿里云维护模型库。V1.x版本的模型依然保存在亚马逊并且将会在未来逐渐弃用。

## 共同设置

- 所有模型都是在`coco_2017_train`上训练，在`coco_2017_val`上测试。
- 我们使用分布式训练。
- 所有pytorch-style的ImageNet预训练主干网络来自PyTorch的模型库，caffe-style的预训练主干网络来自detectron2最新开源的模型。
- 为了与其他代码库公平比较，文档中所写的GPU内存是8个GPU的`torch.cuda.max_memory_allocated()`的最大值，此值通常小于 nvidia-smi 显示的值。 
- 我们所写的模型推理时间包含网络前向时间和后处理时间，不包含数据加载时间。所有结果通过[benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py)脚本计算所得。该脚本会计算推理2000张图像的平均时间。
