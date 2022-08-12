exp_name=yolov3_d53_320_273e_coco_ipu
num_replicas=1
python3 tools/train.py configs/yolo/${exp_name}.py --ipu-replicas $num_replicas --no-validate &&
python3 tools/test.py configs/yolo/${exp_name}.py work_dirs/${exp_name}/latest.pth --work-dir work_dirs/${exp_name}/ --eval bbox --ipu-replicas 1
