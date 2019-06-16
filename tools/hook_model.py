import os
import mmcv

from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import get_dataset, build_dataloader


def forward_hook(module, data_input, data_output):
    """register_forward_hook(hook)"""
    print(data_input.data.shape)
    print(data_output.data.shape)


def backward_hook(module, grad_input, grad_output):
    """register_backward_hook(hook)"""
    print(grad_input.data.shape)
    print(grad_output.data.shape)


HOOT_MODE = "train"  # "inference" or "train"
ROOT_DIR = '/home/hellcatzm/mmdetection'
CONFIG_NAME = 'configs/carbonate/mask_rcnn_r101_fpn_1x.py'

config_file = os.path.join(ROOT_DIR, CONFIG_NAME)
cfg = mmcv.Config.fromfile(config_file)
checkpoint_file = os.path.join(os.path.join(ROOT_DIR, cfg.work_dir), 'latest.pth')

if HOOT_MODE == "inference":
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = 'data/carbonate/val_images/carbonate23.tif'

    #_____________________________________________________________________
    """
    在感兴趣的层注册钩子查看数据流
    """
    # _____________________________________________________________________

    result = inference_detector(model, img)
elif HOOT_MODE == "train":
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

    cfg.data.train.ann_file = os.path.join(ROOT_DIR,
                                           cfg.data.train.ann_file)
    cfg.data.train.img_prefix = os.path.join(ROOT_DIR,
                                             cfg.data.train.img_prefix)
    dataset = get_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        dist=False)
    model.CLASSES = dataset.CLASSES
    batch_data = next(iter(dataloader))

    # _____________________________________________________________________
    """
    在感兴趣的层注册钩子查看数据流
    """
    # _____________________________________________________________________

    losses = model(img=batch_data['img'].data[0].cuda(),
                   img_meta=batch_data['img_meta'].data[0],
                   gt_bboxes=[t.cuda() for t in batch_data['gt_bboxes'].data[0]],
                   gt_labels=[t.cuda() for t in batch_data['gt_labels'].data[0]],
                   gt_bboxes_ignore=batch_data['gt_bboxes_ignore'].data[0],
                   gt_masks=[t for t in batch_data['gt_masks'].data[0]])  # 传入numpy数组即可

