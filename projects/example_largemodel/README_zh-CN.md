# 视觉大模型实践案例

本工程用于探索如何在消费级显卡上成功训练相对大的视觉模型。

虽然视觉模型并没有像 LLM 那样有极其夸张的参数量，但是即使常用的以 Swin Large 为 backbone 的模型，都需要在 A100 上才能成功训练，这无疑阻碍了用户在视觉大模型上的探索和实验。因此本工程将探索在 3090 等 24G 甚至更小显存的消级显卡上如何训练视觉大模型。

本工程主要涉及到的训练技术有 `FSDP`、`DeepSpeed` 和 `ColossalAI` 等常用大模型训练技术。

本工程将不断更新完善，如果你有比较好的探索和意见，也非常欢迎提 PR

## 依赖

```text
mmengine >=0.9.0 # 案例 1
deepspeed # 案例 2
fairscale # 案例 2
```

## 案例 1： 采用 8 张 24G 3090 显卡结合 FSDP 训练 `dino-5scale_swin-l_fsdp_8xb2-12e_coco.py`

```bash
cd mmdetection
./tools/dist_train.sh projects/example_largemodel/dino-5scale_swin-l_fsdp_8xb2-12e_coco.py 8
./tools/dist_train.sh projects/example_largemodel/dino-5scale_swin-l_fsdp_8xb2-12e_coco.py 8 --amp
```

| ID  | AMP | GC of Backbone | GC of Encoder | FSDP | Peak Mem (GB) | Iter Time (s) |
| :-: | :-: | :------------: | :-----------: | :--: | :-----------: | :-----------: |
|  1  |     |                |               |      |   49 (A100)   |      0.9      |
|  2  |  √  |                |               |      |   39 (A100)   |      1.2      |
|  3  |     |       √        |               |      |   33 (A100)   |      1.1      |
|  4  |  √  |       √        |               |      |   25 (A100)   |      1.3      |
|  5  |     |       √        |       √       |      |      18       |      2.2      |
|  6  |  √  |       √        |       √       |      |      13       |      1.6      |
|  7  |     |       √        |       √       |  √   |      14       |      2.9      |
|  8  |  √  |       √        |       √       |  √   |      8.5      |      2.4      |

- AMP: 混合精度训练
- GC: 梯度/激活值检查点
- FSDP: ZeRO-3 结合梯度检查点
- Iter Time: 一次迭代训练总时间

从上表可以看出：

1. 采用 FSDP 结合 AMP 和 GC 技术，可以将最初的 49G 显存降低为 8.5G，但是会增加 1.7 倍训练时间
2. 在目标检测视觉模型中，占据最大显存的是激活值，而不是优化器状态，这和 LLM 不同，因此用户应该首选梯度检查点，而不是 FSDP
3. 如果不开启梯度检查点，仅开启 FSDP 的话依然会 OOM，即使尝试了更加细致的参数切分策略
4. 虽然 AMP 可以减少不少显存，但是有些算法使用 AMP 会导致精度下降而 FSDP 不会

## 案例 2： 采用 8 张 24G 3090 显卡结合 DeepSpeed 训练 `dino-5scale_swin-l_deepspeed_8xb2-12e_coco.py`

```bash
cd mmdetection
./tools/dist_train.sh projects/example_largemodel/dino-5scale_swin-l_deepspeed_8xb2-12e_coco.py 8
```

很遗憾，到目前为止这依然是一个失败的案例，因为梯度始终会溢出导致精度很低。

| ID  | AMP | GC of Backbone | GC of Encoder | DeepSpeed | Peak Mem (GB) | Iter Time (s) |
| :-: | :-: | :------------: | :-----------: | :-------: | :-----------: | :-----------: |
|  1  |     |                |               |           |   49 (A100)   |      0.9      |
|  2  |  √  |                |               |           |   39 (A100)   |      1.2      |
|  3  |  √  |       √        |               |           |   25 (A100)   |      1.3      |
|  4  |  √  |       √        |               |     √     |     10.5      |      1.5      |
|  5  |  √  |       √        |       √       |           |      13       |      1.6      |
|  6  |  √  |       √        |       √       |     √     |      5.0      |      1.4      |

从上表可以看出：

1. DeepSpeed 易用性上相比于 FSDP 有很大提升，因为梯度检查点可以用 torch 原生的而不需要修改特殊定制，同时也没有 `auto_wrap_policy` 这个需要用户自行设置的参数
2. DeepSpeed ZeRO 系列必须要采用 FP16 模式，其底层是采用了 NVIDIA’s Apex package, 其使用 Apex 的 AMP O2 模式，这导致需要修改代码，并且 O2 模式采用大量 FP16 计算导致 DINO 算法无法正常训练，但是它的这种模式可以显著节省显存，相比于 torch 官方的 AMP，类型转换更加彻底

从上述分析可知，如果 DeepSpeed 能够在不降低性能情况下成功训练 DINO 模型，那么其将比 FSDP 具备比较大的优势。如果您对 DeepSpeed 和 Apex 有比较深入的了解同时有兴趣排查精度问题，欢迎反馈或者提 PR

前面说过由于 Apex AMP O2 的特殊性，目前的 MMDetection 无法训练 DINO 模型，考虑到这是一个失败的案例，因此将修改的代码放在了 https://github.com/hhaAndroid/mmdetection/tree/dino_deepspeed 分支，其对应修改见 [commit](https://github.com/hhaAndroid/mmdetection/commit/0c825ae38e2cee3d11a20c5c4adf24ee682d0a55)。如果您有兴趣尝试，可以拉取该分支进行试验。
