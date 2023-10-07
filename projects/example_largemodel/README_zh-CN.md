# 视觉大模型实践案例

本工程用于探索如何在消费级显卡上成功训练相对大的视觉模型。

虽然视觉模型并没有像 LLM 那样有极其夸张的参数量，但是即使常用的以 Swin Large 为 backbone 的模型，都需要在 A100 上才能成功训练，这无疑阻碍了用户在视觉大模型上的探索和实验。因此本工程将探索在 3090 等 24G 甚至更小显存的消级显卡上如何训练视觉大模型。

本工程主要涉及到的训练技术有 `FSDP`、`DeepSpeed` 和 `ColossalAI` 等常用大模型训练技术。

本工程将不断更新完善，如果你有比较好的探索和意见，也非常欢迎提 PR

## 案例 1： 采用 8张 24G 3090 显卡结合 FSDP 训练 `dino-5scale_swin-l_fsdp_8xb2-12e_coco.py`

| ID | AMP | GC  | FSDP | Mem | Time |
|:--:| :-: | :-: | :--: | :-: | :--: |
| 1  |     |     |      |     |      |
| 2  |  √  |     |      |     |      |
| 3  |     |  √  |      |     |      |
| 4  |  √  |  √  |      |     |      |
| 5  |     |     |  √   |     |      |
| 6  |     |  √  |  √   |     |      |
| 7  |  √  |  √  |  √   |     |      |

- AMP: Automatic Mixed Precision
- GC: Gradient Checkpointing
- FSDP: ZeRO-3
- Time: Total training time for one iteration.

