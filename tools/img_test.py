import mmcv
from mmcv import Config
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_detector, inference_detector, show_result



config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = 'data/carbonate/val_images/carbonate23.tif'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)


