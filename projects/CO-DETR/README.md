# CO-DETR

> [DETRs with Collaborative Hybrid Assignments Training](https://arxiv.org/abs/2211.12860)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we provide the observation that too few queries assigned as positive samples in DETR with one-to-one set matching leads to sparse supervision on the encoder's output which considerably hurt the discriminative feature learning of the encoder and vice visa for attention learning in the decoder. To alleviate this, we present a novel collaborative hybrid assignments training scheme, namely o-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. This new training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training the multiple parallel auxiliary heads supervised by one-to-many label assignments such as ATSS and Faster RCNN. In addition, we conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve the training efficiency of positive samples in the decoder. In inference, these auxiliary heads are discarded and thus our method introduces no additional parameters and computational cost to the original detector while requiring no hand-crafted non-maximum suppression (NMS). We conduct extensive experiments to evaluate the effectiveness of the proposed approach on DETR variants, including DAB-DETR, Deformable-DETR, and DINO-Deformable-DETR. The state-of-the-art DINO-Deformable-DETR with Swin-L can be improved from 58.5% to 59.5% AP on COCO val. Surprisingly, incorporated with ViT-L backbone, we achieve 66.0% AP on COCO test-dev and 67.9% AP on LVIS val, outperforming previous methods by clear margins with much fewer model sizes.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/dceaf7ee-cd6c-4be0-b7b1-5b01a7f11724"/>
</div>

## Results and Models

|   Model   | Backbone | Epochs | Aug  | Dataset | Mem (GB) | box AP | Config | Download |
| :-------: | :------: | :----: | :--: | :-----: | :------: | :----: | :----: | :------: |
|  Co-DINO  |   R50    |   12   | LSJ  |  COCO   |          |  52.1  |        |          |
| Co-DINO\* |   R50    |   12   | DETR |  COCO   |          |  52.1  |        |          |
| Co-DINO\* |   R50    |   36   | LSJ  |  COCO   |          |  54.8  |        |          |
| Co-DINO\* |  Swin-L  |   12   | DETR |  COCO   |          |  58.9  |        |          |
| Co-DINO\* |  Swin-L  |   12   | LSJ  |  COCO   |          |  59.3  |        |          |
| Co-DINO\* |  Swin-L  |   36   | DETR |  COCO   |          |  60.0  |        |          |
| Co-DINO\* |  Swin-L  |   36   | LSJ  |  COCO   |          |  60.7  |        |          |

Note

- Models labeled * are not trained by us, but from [CO-DETR](https://github.com/Sense-X/Co-DETR) official website.
- We find that the performance is unstable and may fluctuate by about 0.3 mAP.
- If you want to save GPU memory by enabling checkpointing, please use the `pip install fairscale` command.
