# 学习配置文件

MMDetection 和其他 OpenMMLab 仓库使用 [MMEngine 的配置文件系统](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)。 配置文件使用了模块化和继承设计，以便于进行各类实验。

## 配置文件的内容

MMDetection 采用模块化设计，所有功能的模块都可以通过配置文件进行配置。 以 Mask R-CNN 为例，我们将根据不同的功能模块介绍配置文件中的各个字段：

### 模型配置

在 mmdetection 的配置中，我们使用 `model` 字段来配置检测算法的组件。 除了 `backbone`、`neck` 等神经网络组件外，还需要 `data_preprocessor`、`train_cfg` 和 `test_cfg`。 `data_preprocessor` 负责对 dataloader 输出的每一批数据进行预处理。 模型配置中的 `train_cfg` 和 `test_cfg` 用于设置训练和测试组件的超参数。

```python
model = dict(
    type='MaskRCNN',  # 检测器名
    data_preprocessor=dict(  # 数据预处理器的配置，通常包括图像归一化和 padding
        type='DetDataPreprocessor',  # 数据预处理器的类型，参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor
        mean=[123.675, 116.28, 103.53],  # 用于预训练骨干网络的图像归一化通道均值，按 R、G、B 排序
        std=[58.395, 57.12, 57.375],  # 用于预训练骨干网络的图像归一化通道标准差，按 R、G、B 排序
        bgr_to_rgb=True,  # 是否将图片通道从 BGR 转为 RGB
        pad_mask=True,  # 是否填充实例分割掩码
        pad_size_divisor=32),  # padding 后的图像的大小应该可以被 ``pad_size_divisor`` 整除
    backbone=dict(  # 主干网络的配置文件
        type='ResNet',  # 主干网络的类别，可用选项请参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.ResNet
        depth=50,  # 主干网络的深度，对于 ResNet 和 ResNext 通常设置为 50 或 101
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入
        out_indices=(0, 1, 2, 3),  # 每个状态产生的特征图输出的索引
        frozen_stages=1,  # 第一个状态的权重被冻结
        norm_cfg=dict(  # 归一化层(norm layer)的配置项
            type='BN',  # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta
        norm_eval=True,  # 是否冻结 BN 里的统计项
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 加载通过 ImageNet 预训练的模型
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN，我们同样支持 'NASFPN', 'PAFPN' 等，更多细节可以参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.FPN
        in_channels=[256, 512, 1024, 2048],  # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,  # 金字塔特征图每一层的输出通道
        num_outs=5),  # 输出的范围(scales)
    rpn_head=dict(
        type='RPNHead',  # rpn_head 的类型是 'RPNHead', 我们也支持 'GARPNHead' 等，更多细节可以参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.dense_heads.RPNHead
        in_channels=256,  # 每个输入特征图的输入通道，这与 neck 的输出通道一致
        feat_channels=256,  # head 卷积层的特征通道
        anchor_generator=dict(  # 锚点(Anchor)生成器的配置
            type='AnchorGenerator',  # 大多数方法使用 AnchorGenerator 作为锚点生成器, SSD 检测器使用 `SSDAnchorGenerator`。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py#L18
            scales=[8],  # 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0],  # 高度和宽度之间的比率
            strides=[4, 8, 16, 32, 64]),  # 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes
        bbox_coder=dict(  # 在训练和测试期间对框进行编码和解码
            type='DeltaXYWHBBoxCoder',  # 框编码器的类别，'DeltaXYWHBBoxCoder' 是最常用的，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py#L13
            target_means=[0.0, 0.0, 0.0, 0.0],  # 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 用于编码和解码框的标准差
        loss_cls=dict(  # 分类分支的损失函数配置
            type='CrossEntropyLoss',  # 分类分支的损失类型，我们也支持 FocalLoss 等，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/cross_entropy_loss.py#L201
            use_sigmoid=True,  # RPN 通常进行二分类，所以通常使用 sigmoid 函数
            los_weight=1.0),  # 分类分支的损失权重
        loss_bbox=dict(  # 回归分支的损失函数配置
            type='L1Loss',  # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/smooth_l1_loss.py#L56
            loss_weight=1.0)),  # 回归分支的损失权重
    roi_head=dict(  # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步
        type='StandardRoIHead',  # RoI head 的类型，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/standard_roi_head.py#L17
        bbox_roi_extractor=dict(  # 用于 bbox 回归的 RoI 特征提取器
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用 SingleRoIExtractor，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L13
            roi_layer=dict(  # RoI 层的配置
                type='RoIAlign',  # RoI 层的类别, 也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack，更多细节请参考 https://mmcv.readthedocs.io/en/latest/api.html#mmcv.ops.RoIAlign
                output_size=7,  # 特征图的输出大小
                sampling_ratio=0),  # 提取 RoI 特征时的采样率。0 表示自适应比率
            out_channels=256,  # 提取特征的输出通道
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅，应该与主干的架构保持一致
        bbox_head=dict(  # RoIHead 中 box head 的配置
            type='Shared2FCBBoxHead',  # bbox head 的类别，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L220
            in_channels=256,  # bbox head 的输入通道。 这与 roi_extractor 中的 out_channels 一致
            fc_out_channels=1024,  # FC 层的输出特征通道
            roi_feat_size=7,  # 候选区域(Region of Interest)特征的大小
            num_classes=80,  # 分类的类别数量
            bbox_coder=dict(  # 第二阶段使用的框编码器
                type='DeltaXYWHBBoxCoder',  # 框编码器的类别，大多数情况使用 'DeltaXYWHBBoxCoder'
                target_means=[0.0, 0.0, 0.0, 0.0],  # 用于编码和解码框的均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 编码和解码的标准差。因为框更准确，所以值更小，常规设置时 [0.1, 0.1, 0.2, 0.2]。
            reg_class_agnostic=False,  # 回归是否与类别无关
            loss_cls=dict(  # 分类分支的损失函数配
                type='CrossEntropyLoss',  # 分类分支的损失类型，我们也支持 FocalLoss 等
                use_sigmoid=False,  # 是否使用 sigmoid
                loss_weight=1.0),  # 分类分支的损失权重
            loss_bbox=dict(  # 回归分支的损失函数配置
                type='L1Loss',  # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等
                loss_weight=1.0)),  # 回归分支的损失权重
        mask_roi_extractor=dict(  # 用于 mask 生成的 RoI 特征提取器
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用 SingleRoIExtractor
            roi_layer=dict(  # 提取实例分割特征的 RoI 层配置
                type='RoIAlign',  # RoI 层的类型，也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack
                output_size=14,  # 特征图的输出大小
                sampling_ratio=0),  # 提取 RoI 特征时的采样率
            out_channels=256,  # 提取特征的输出通道
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅
        mask_head=dict(  # mask 预测 head 模型
            type='FCNMaskHead',  # mask head 的类型，更多细节请参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.roi_heads.FCNMaskHead
            num_convs=4,  # mask head 中的卷积层数
            in_channels=256,  # 输入通道，应与 mask roi extractor 的输出通道一致
            conv_out_channels=256,  # 卷积层的输出通道
            num_classes=80,  # 要分割的类别数
            loss_mask=dict(  # mask 分支的损失函数配置
                type='CrossEntropyLoss',  # 用于分割的损失类型
                use_mask=True,  # 是否只在正确的类中训练 mask
                loss_weight=1.0))),  # mask 分支的损失权重
    train_cfg = dict(  # rpn 和 rcnn 训练超参数的配置
        rpn=dict(  # rpn 的训练配置
            assigner=dict(  # 分配器(assigner)的配置
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 用于许多常见的检测器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14
                pos_iou_thr=0.7,  # IoU >= 0.7(阈值) 被视为正样本
                neg_iou_thr=0.3,  # IoU < 0.3(阈值) 被视为负样本
                min_pos_iou=0.3,  # 将框作为正样本的最小 IoU 阈值
                match_low_quality=True,  # 是否匹配低质量的框(更多细节见 API 文档)
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict(  # 正/负采样器(sampler)的配置
                type='RandomSampler',  # 采样器类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14
                num=256,  # 样本数量。
                pos_fraction=0.5,  # 正样本占总样本的比例
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限
                add_gt_as_proposals=False),  # 采样后是否添加 GT 作为 proposal
            allowed_border=-1,  # 填充有效锚点后允许的边框
            pos_weight=-1,  # 训练期间正样本的权重
            debug=False),  # 是否设置调试(debug)模式
        rpn_proposal=dict(  # 在训练期间生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于 `GARPNHead` ，naive rpn 不支持 nms cross levels
            nms_pre=2000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在 GARPNHHead 中起作用
            max_per_img=1000,  # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类别
                iou_threshold=0.7 # NMS 的阈值
                ),
            min_bbox_size=0),  # 允许的最小 box 尺寸
        rcnn=dict(  # roi head 的配置。
            assigner=dict(  # 第二阶段分配器的配置，这与 rpn 中的不同
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 目前用于所有 roi_heads。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14
                pos_iou_thr=0.5,  # IoU >= 0.5(阈值)被认为是正样本
                neg_iou_thr=0.5,  # IoU < 0.5(阈值)被认为是负样本
                min_pos_iou=0.5,  # 将 box 作为正样本的最小 IoU 阈值
                match_low_quality=False,  # 是否匹配低质量下的 box(有关更多详细信息，请参阅 API 文档)
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict(
                type='RandomSampler',  # 采样器的类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14
                num=512,  # 样本数量
                pos_fraction=0.25,  # 正样本占总样本的比例
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限
                add_gt_as_proposals=True
            ),  # 采样后是否添加 GT 作为 proposal
            mask_size=28,  # mask 的大小
            pos_weight=-1,  # 训练期间正样本的权重
            debug=False)),  # 是否设置调试模式
    test_cfg = dict(  # 用于测试 rpn 和 rcnn 超参数的配置
        rpn=dict(  # 测试阶段生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于 `GARPNHead`，naive rpn 不支持做 NMS cross levels
            nms_pre=1000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在 `GARPNHHead` 中起作用
            max_per_img=1000,  # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类型
                iou_threshold=0.7 # NMS 阈值
                ),
            min_bbox_size=0),  # box 允许的最小尺寸
        rcnn=dict(  # roi heads 的配置
            score_thr=0.05,  # bbox 的分数阈值
            nms=dict(  # 第二步的 NMS 配置
                type='nms',  # NMS 的类型
                iou_thr=0.5),  # NMS 的阈值
            max_per_img=100,  # 每张图像的最大检测次数
            mask_thr_binary=0.5)))  # mask 预处的阈值
```

### 数据集和评测器配置

在使用[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html) 进行训练、测试、验证时，我们需要配置 [Dataloader](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/dataset.html)。构建数据 dataloader 需要设置数据集（dataset）和数据处理流程（data pipeline）。 由于这部分的配置较为复杂，我们使用中间变量来简化 dataloader 配置的编写。

```python
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'data/coco/'  # 数据的根路径。

train_pipeline = [  # 训练数据处理流程
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='LoadAnnotations',  # 第 2 个流程，对于当前图像，加载它的注释信息。
        with_bbox=True,  # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
        with_mask=True,  # 是否使用 instance mask，实例分割需要设置为 True。
        poly2mask=False),  # 是否将 polygon mask 转化为 instance mask, 设置为 False 以加速和节省内存。
    dict(
        type='Resize',  # 变化图像和其标注大小的流程。
        scale=(1333, 800),  # 图像的最大尺寸
        keep_ratio=True  # 是否保持图像的长宽比。
        ),
    dict(
        type='RandomFlip',  # 翻转图像和其标注的数据增广流程。
        prob=0.5),  # 翻转图像的概率。
    dict(type='PackDetInputs')  # 将数据转换为检测器输入格式的流程
]
test_pipeline = [  # 测试数据处理流程
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 变化图像大小的流程。
    dict(
        type='PackDetInputs',  # 将数据转换为检测器输入格式的流程
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(  # 训练 dataloader 配置
    batch_size=2,  # 单个 GPU 的 batch size
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。请参考 https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # 标注文件路径
        data_prefix=dict(img='train2017/'),  # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 图片和标注的过滤配置
        pipeline=train_pipeline))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。
val_dataloader = dict(  # 验证 dataloader 配置
    batch_size=1,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,  # 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # 测试 dataloader 配置
```

[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html) 用于计算训练模型在验证和测试数据集上的指标。评测器的配置由一个或一组评价指标（Metric）配置组成：

```python
val_evaluator = dict(  # 验证过程使用的评测器
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json',  # 标注文件路径
    metric=['bbox', 'segm'],  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)
test_evaluator = val_evaluator  # 测试过程使用的评测器
```

由于测试数据集没有标注文件，因此 MMDetection 中的 test_dataloader 和 test_evaluator 配置通常等于val。 如果要保存在测试数据集上的检测结果，则可以像这样编写配置：

```python
# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    metric=['bbox', 'segm'],
    format_only=True,  # 只将模型输出转换为 coco 的 JSON 格式并保存
    outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀
```

### 训练和测试的配置

MMEngine 的 Runner 使用 Loop 来控制训练，验证和测试过程。
用户可以使用这些字段设置最大训练轮次和验证间隔。

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型
```

### 优化相关配置

`optim_wrapper` 是配置优化相关设置的字段。优化器封装（OptimWrapper）不仅提供了优化器的功能，还支持梯度裁剪、混合精度训练等功能。更多内容请看[优化器封装教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) 。

```python
optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.02,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )
```

`param_scheduler` 字段用于配置参数调度器（Parameter Scheduler）来调整优化器的超参数（例如学习率和动量）。 用户可以组合多个调度器来创建所需的参数调整策略。 在 [参数调度器教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 和 [参数调度器 API 文档](https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.optim._ParamScheduler.html#mmengine.optim._ParamScheduler) 中查找更多信息。

```python
param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]
```

### 钩子配置

用户可以在训练、验证和测试循环上添加钩子，以便在运行期间插入一些操作。配置中有两种不同的钩子字段，一种是 `default_hooks`，另一种是 `custom_hooks`。

`default_hooks` 是一个字典，用于配置运行时必须使用的钩子。这些钩子具有默认优先级，如果未设置，runner 将使用默认值。如果要禁用默认钩子，用户可以将其配置设置为 `None`。更多内容请看 [钩子教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html) 。

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
```

`custom_hooks` 是一个列表。用户可以在这个字段中加入自定义的钩子。

```python
custom_hooks = []
```

### 运行相关配置

```python
default_scope = 'mmdet'  # 默认的注册器域名，默认从此注册器域中寻找模块。请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程设置
        mp_start_method='fork',  # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。请参考 https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),  # 分布式相关设置
)

vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端，请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor',  # 日志处理器用于处理运行时日志
    window_size=50,  # 日志数值的平滑窗口
    by_epoch=True)  # 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。

log_level = 'INFO'  # 日志等级
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
```

## Iter-based 配置

MMEngine 的 Runner 除了基于轮次的训练循环（epoch）外，还提供了基于迭代（iteration）的训练循环。
要使用基于迭代的训练，用户应该修改 `train_cfg`、`param_scheduler`、`train_dataloader`、`default_hooks` 和 `log_processor`。
以下是将基于 epoch 的 RetinaNet 配置更改为基于 iteration 的示例：configs/retinanet/retinanet_r50_fpn_90k_coco.py

```python
# iter-based 训练配置
train_cfg = dict(
    _delete_=True,  # 忽略继承的配置文件中的值（可选）
    type='IterBasedTrainLoop',  # iter-based 训练循环
    max_iters=90000,  # 最大迭代次数
    val_interval=10000)  # 每隔多少次进行一次验证


# 将参数调度器修改为 iter-based
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# 切换至 InfiniteSampler 来避免 dataloader 重启
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))

# 将模型检查点保存间隔设置为按 iter 保存
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))

# 将日志格式修改为 iter-based
log_processor = dict(by_epoch=False)
```

## 配置文件继承

在 `config/_base_` 文件夹下有 4 个基本组件类型，分别是：数据集(dataset)，模型(model)，训练策略(schedule)和运行时的默认设置(default runtime)。许多方法，例如 Faster R-CNN、Mask R-CNN、Cascade R-CNN、RPN、SSD 能够很容易地构建出来。由 `_base_` 下的组件组成的配置，被我们称为 _原始配置(primitive)_。

对于同一文件夹下的所有配置，推荐**只有一个**对应的**原始配置**文件。所有其他的配置文件都应该继承自这个**原始配置**文件。这样就能保证配置文件的最大继承深度为 3。

为了便于理解，我们建议贡献者继承现有方法。例如，如果在 Faster R-CNN 的基础上做了一些修改，用户首先可以通过指定 `_base_ = ../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` 来继承基础的 Faster R-CNN 结构，然后修改配置文件中的必要参数以完成继承。

如果你在构建一个与任何现有方法不共享结构的全新方法，那么可以在 `configs` 文件夹下创建一个新的例如 `xxx_rcnn` 文件夹。

更多细节请参考 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html) 。

通过设置 `_base_` 字段，我们可以设置当前配置文件继承自哪些文件。

当 `_base_` 为文件路径字符串时，表示继承一个配置文件的内容。

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'
```

当 `_base_` 是多个文件路径的列表时，表示继承多个文件。

```python
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

如果需要检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

### 忽略基础配置文件里的部分内容

有时，您也许会设置 `_delete_=True` 去忽略基础配置文件里的一些域内容。 您也许可以参照 [MMEngine 配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html) 来获得一些简单的指导。

在 MMDetection 里，例如为了改变  Mask R-CNN 的主干网络的某些内容：

```python
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(...),
    rpn_head=dict(...),
    roi_head=dict(...))
```

基础配置的 `Mask R-CNN` 使用 `ResNet-50`，在需要将主干网络改成 `HRNet` 的时候，因为 `HRNet` 和 `ResNet` 中有不同的字段，需要使用 `_delete_=True` 将新的键去替换 `backbone` 域内所有老的键。

```python
_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')),
    neck=dict(...))
```

### 使用配置文件里的中间变量

配置文件里会使用一些中间变量，例如数据集里的 `train_pipeline`/`test_pipeline`。我们在定义新的 `train_pipeline`/`test_pipeline` 之后，需要将它们传递到 `data` 里。例如，我们想在训练或测试时，改变 Mask R-CNN 的多尺度策略 (multi scale strategy)，`train_pipeline`/`test_pipeline` 是我们想要修改的中间变量。

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

我们首先定义新的 `train_pipeline`/`test_pipeline` 然后传递到 `data` 里。

同样的，如果我们想从 `SyncBN` 切换到 `BN` 或者 `MMSyncBN`，我们需要修改配置文件里的每一个  `norm_cfg`。

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```

### 复用 \_base\_ 文件中的变量

如果用户希望在当前配置中复用 base 文件中的变量，则可以通过使用 `{{_base_.xxx}}` 的方式来获取对应变量的拷贝。例如：

```python
_base_ = './mask-rcnn_r50_fpn_1x_coco.py'

a = {{_base_.model}}  # 变量 a 等于 _base_ 中定义的 model
```

## 通过脚本参数修改配置

当运行 `tools/train.py` 和 `tools/test.py` 时，可以通过 `--cfg-options` 来修改配置文件。

- 更新字典链中的配置

  可以按照原始配置文件中的 dict 键顺序地指定配置预选项。例如，使用 `--cfg-options model.backbone.norm_eval=False` 将模型主干网络中的所有 BN 模块都改为 `train` 模式。

- 更新配置列表中的键

  在配置文件里，一些字典型的配置被包含在列表中。例如，数据训练流程 `data.train.pipeline` 通常是一个列表，比如 `[dict(type='LoadImageFromFile'), ...]`。如果需要将 `'LoadImageFromFile'` 改成 `'LoadImageFromWebcam'`，需要写成下述形式： `--cfg-options data.train.pipeline.0.type=LoadImageFromNDArray`.

- 更新列表或元组的值

  如果要更新的值是列表或元组。例如，配置文件通常设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`. 如果需要改变这个键，可以通过 `--cfg-options model.data_preprocessor.mean="[127,127,127]"` 来重新设置。需要注意，引号 " 是支持列表或元组数据类型所必需的，并且在指定值的引号内**不允许**有空格。

## 配置文件名称风格

我们遵循以下样式来命名配置文件。建议贡献者遵循相同的风格。

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

文件名分为五个部分。 每个部分用`_`连接，每个部分内的单词应该用`-`连接。

- `{algorithm name}`: 算法的名称。 它可以是检测器名称，例如 `faster-rcnn`、`mask-rcnn` 等。也可以是半监督或知识蒸馏算法，例如 `soft-teacher`、`lad` 等等
- `{component names}`: 算法中使用的组件名称，如 backbone、neck 等。例如 `r50-caffe_fpn_gn-head` 表示在算法中使用 caffe 版本的 ResNet50、FPN 和 使用了 Group Norm 的检测头。
- `{training settings}`: 训练设置的信息，例如 batch 大小、数据增强、损失、参数调度方式和训练最大轮次/迭代。 例如：`4xb4-mixup-giou-coslr-100e` 表示使用 8 个 gpu 每个 gpu 4 张图、mixup 数据增强、GIoU loss、余弦退火学习率，并训练 100 个 epoch。
  缩写介绍:
  - `{gpu x batch_per_gpu}`: GPU 数和每个 GPU 的样本数。`bN` 表示每个 GPU 上的 batch 大小为 N。例如 `4x4b` 是 4 个 GPU 每个 GPU 4 张图的缩写。如果没有注明，默认为 8 卡每卡 2 张图。
  - `{schedule}`: 训练方案，选项是 `1x`、 `2x`、 `20e` 等。`1x`  和 `2x` 分别代表 12 epoch 和 24 epoch，`20e` 在级联模型中使用，表示 20 epoch。对于 `1x`/`2x`，初始学习率在第 8/16 和第 11/22 epoch 衰减 10 倍；对于 `20e` ，初始学习率在第 16 和第 19 epoch 衰减 10 倍。
- `{training dataset information}`: 训练数据集，例如 `coco`, `coco-panoptic`, `cityscapes`, `voc-0712`, `wider-face`。
- `{testing dataset information}` (可选): 测试数据集，用于训练和测试在不同数据集上的模型配置。 如果没有注明，则表示训练和测试的数据集类型相同。
