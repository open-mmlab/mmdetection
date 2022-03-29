import pytest
import tempfile
import os.path as osp
from copy import deepcopy 
from mmcv.utils import Config
from mmdet.utils import replace_config

def test_replace_config():
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write('configs')

    ori_cfg_dict = dict()
    ori_cfg_dict["work_dir"] = "work_dirs/${cfg_name}/${percent}/${fold}"
    ori_cfg_dict["percent"] = 5
    ori_cfg_dict["fold"] = 1
    ori_cfg_dict["semi_wrapper"] = dict(type="SoftTeacher", model="${model}")
    ori_cfg_dict["model"] = dict(
        type='FasterRCNN', 
        backbone=dict(type='ResNet'), 
        neck=dict(type='FPN'),
        rpn_head=dict(type='RPNHead'),
        roi_head=dict(type='StandardRoIHead'),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type='MaxIoUAssigner'),
                sampler=dict(type='RandomSampler'),
            ),
            rpn_proposal=dict(nms=dict(type='nms', iou_threshold=0.7)),
            rcnn=dict(
                assigner=dict(type='MaxIoUAssigner'),
                sampler=dict(type='RandomSampler'),
            ),
        ),
        test_cfg=dict(
            rpn=dict(nms=dict(type='nms', iou_threshold=0.7)),
            rcnn=dict(nms=dict(type='nms', iou_threshold=0.5)),
        ),
    )
    ori_cfg_dict['iou_threshold'] = dict(
        rpn_proposal_nms = "${model.train_cfg.rpn_proposal.nms.iou_threshold}",
        test_rpn_nms = "${model.test_cfg.rpn.nms.iou_threshold}",
        test_rcnn_nms = "${model.test_cfg.rcnn.nms.iou_threshold}",
    )
    
    ori_cfg = Config(ori_cfg_dict, filename=config_path)
    replaced_cfg = replace_config(deepcopy(ori_cfg))

    assert replaced_cfg.work_dir == f"work_dirs/{osp.basename(temp_file.name)}/5/1"
    assert replaced_cfg.model.model == ori_cfg.model
    assert replaced_cfg.iou_threshold.rpn_proposal_nms == ori_cfg.model.train_cfg.rpn_proposal.nms.iou_threshold
    