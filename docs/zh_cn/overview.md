# 概述

本章向您介绍 MMDetection 的整体框架，并提供详细的教程链接。

## 什么是 MMDetection

![图片](https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png)

MMDetection 是一个目标检测工具箱，包含了丰富的目标检测、实例分割、全景分割算法以及相关的组件和模块，下面是它的整体框架：

MMDetection 由 7 个主要部分组成，apis、structures、datasets、models、engine、evaluation 和 visualization。

- **apis** 为模型推理提供高级 API。
- **structures** 提供 bbox、mask 和 DetDataSample 等数据结构。
- **datasets** 支持用于目标检测、实例分割和全景分割的各种数据集。
  - **transforms** 包含各种数据增强变换。
  - **samplers** 定义了不同的数据加载器采样策略。
- **models** 是检测器最重要的部分，包含检测器的不同组件。
  - **detectors** 定义所有检测模型类。
  - **data_preprocessors** 用于预处理模型的输入数据。
  - **backbones** 包含各种骨干网络。
  - **necks** 包含各种模型颈部组件。
  - **dense_heads** 包含执行密集预测的各种检测头。
  - **roi_heads** 包含从 RoI 预测的各种检测头。
  - **seg_heads** 包含各种分割头。
  - **losses** 包含各种损失函数。
  - **task_modules** 为检测任务提供模块，例如 assigners、samplers、box coders 和 prior generators。
  - **layers** 提供了一些基本的神经网络层。
- **engine** 是运行时组件的一部分。
  - **runner** 为 [MMEngine 的执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)提供扩展。
  - **schedulers** 提供用于调整优化超参数的调度程序。
  - **optimizers** 提供优化器和优化器封装。
  - **hooks** 提供执行器的各种钩子。
- **evaluation** 为评估模型性能提供不同的指标。
- **visualization** 用于可视化检测结果。

## 如何使用本指南

以下是 MMDetection 的详细指南：

1. 安装说明见[开始你的第一步](get_started.md)。

2. MMDetection 的基本使用方法请参考以下教程。

   - [训练和测试](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/index.html#train-test)

   - [实用工具](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/index.html#useful-tools)

3. 参考以下教程深入了解：

   - [基础概念](https://mmdetection.readthedocs.io/zh_CN/latest/advanced_guides/index.html#basic-concepts)
   - [组件定制](https://mmdetection.readthedocs.io/zh_CN/latest/advanced_guides/index.html#component-customization)

4. 对于 MMDetection 2.x 版本的用户，我们提供了[迁移指南](./migration/migration.md)，帮助您完成新版本的适配。
