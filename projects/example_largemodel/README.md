# Vision Large Model Example

The project is used to explore how to successfully train relatively large visual models on consumer-level graphics cards.

Although the visual model does not have such an exaggerated number of parameters as LLM, even the commonly used models with Swin Large as the backbone need to be trained successfully on A100, which undoubtedly hinders users' exploration and experiments on visual large models. Therefore, this project will explore how to train visual large models on 3090 and even smaller graphics cards with 24G or less memory.

The project mainly involves training technologies such as `FSDP`, `DeepSpeed` and `ColossalAI` commonly used in large model training.

The project will be continuously updated and improved. If you have better exploration and suggestions, you are also welcome to submit a PR

## requirements

```text
mmengine >=0.9.0 # Example 1
deepspeed # Example 2
fairscale # Example 2
```

## Example 1: Train `dino-5scale_swin-l_fsdp_8xb2-12e_coco.py` with 8 24G 3090 GPUs and FSDP

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

- AMP: Automatic Mixed Precision
- GC: Gradient/Activation checkpointing
- FSDP: ZeRO-3 with Activation Checkpointing ZeRO-3
- Iter Time: Total training time for one iteration

From the above analysis, it can be seen that:

1. By combining FSDP with AMP and GC techniques, the initial 49GB of GPU memory can be reduced to 8.5GB, but it comes at the cost of a 1.7x increase in training time.
2. In object detection visual models, the largest memory consumption is due to activation values, rather than optimizer states, which is different from LLM. Therefore, users should prefer gradient checkpoints over FSDP.
3. If gradient checkpoints are not enabled and only FSDP is used, out-of-memory (OOM) errors can still occur, even with more fine-grained parameter splitting strategies.
4. While AMP can significantly reduce memory usage, some algorithms may experience a decrease in precision when using AMP, whereas FSDP does not exhibit this issue.

## Example 2: Train `dino-5scale_swin-l_deepspeed_8xb2-12e_coco.py` with 8 24G 3090 GPUs and DeepSpeed

```bash
cd mmdetection
./tools/dist_train.sh projects/example_largemodel/dino-5scale_swin-l_deepspeed_8xb2-12e_coco.py 8
```

It is a pity that this is still a failed case so far, because the gradient will always overflow, resulting in very low accuracy.

| ID  | AMP | GC of Backbone | GC of Encoder | DeepSpeed | Peak Mem (GB) | Iter Time (s) |
| :-: | :-: | :------------: | :-----------: | :-------: | :-----------: | :-----------: |
|  1  |     |                |               |           |   49 (A100)   |      0.9      |
|  2  |  √  |                |               |           |   39 (A100)   |      1.2      |
|  3  |  √  |       √        |               |           |   25 (A100)   |      1.3      |
|  4  |  √  |       √        |               |     √     |     10.5      |      1.5      |
|  5  |  √  |       √        |       √       |           |      13       |      1.6      |
|  6  |  √  |       √        |       √       |     √     |      5.0      |      1.4      |

From the above analysis, it can be seen that:

1. DeepSpeed has greatly improved usability compared to FSDP. Gradient checkpointing can be done using the native torch functionality without the need for custom modifications, and there is no need for the `auto_wrap_policy` parameter that needs to be set by the user.
2. The DeepSpeed ZeRO series requires the use of FP16 mode and utilizes NVIDIA's Apex package. It uses Apex's AMP O2 mode, which requires code modifications. However, the O2 mode uses a significant amount of FP16 computation, which prevents DINO algorithm from training properly. But this mode can significantly save GPU memory and provides more thorough type conversion compared to torch's official AMP.

From the above analysis, it can be concluded that if DeepSpeed can successfully train the DINO model without reduce performance, it will have a significant advantage over FSDP. If you have a deep understanding of DeepSpeed and Apex and are interested in troubleshooting accuracy issues, your feedback or PR is welcome.

As mentioned earlier, due to the specific nature of Apex AMP O2, the current version of MMDetection cannot train the DINO model. Considering this as a failed case, the modified code has been placed in the [dino_deepspeed branch](https://github.com/hhaAndroid/mmdetection/tree/dino_deepspeed). The corresponding modifications can be seen in this [commit](https://github.com/hhaAndroid/mmdetection/commit/0c825ae38e2cee3d11a20c5c4adf24ee682d0a55). If you are interested, you can pull this branch and experiment with it.
