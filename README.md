## Requirements
Tested on the following environments
- mmdetection==2.26.0
- torch==1.11.0
- cuda 11.3


## Structures
- Original FasterRCNN
  - detectors: "Faster_RCNN" ($mmdet/models/detectors/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "StandardRoIHead" ($mmdet/models/roi_heads/standard_roi_head.py)
  - data: "CocoDataset" ($mmdet/datasets/coco.py)

- MSDET FasterRCNN
  - detectors: "Faster_RCNN_TS" ($msdet/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "ContRoIHead" ($msdet/roi_heads.py)
  - data: "CocoConDataset" ($msdet/coco.py)


## Train and Evaluation
- Train
  - Train with Multi GPUs
    ```
    CUDA_VISIBLE_DEVICES=$GPU_IDs  python -m torch.distributed.launch \
                                          --nproc_per_node=$NUM_GPUs \
                                          --master_port $PORT_NUM \
                                    tools/train.py \
                                          --config $CONFIG_PATH \
                                          --seed $SEED_NUM \
                                          --work-dir $SAVE_DIR \
                                          --launcher pytorch
    ```

  - Train with Single GPU
    ```
    python train.py --gpu-id $GPU_ID \
                    --config $CONFIG_PATH \
                    --seed $SEED_NUM \
                    --work-dir $SAVE_DIR
    ```