import os.path as osp
import tempfile
from copy import deepcopy

import pytest
from mmcv.utils import Config

from mmdet.utils import replace_cfg_vals


def test_replace_cfg_vals():
    temp_file = tempfile.NamedTemporaryFile()
    cfg_path = f'{temp_file.name}.py'
    with open(cfg_path, 'w') as f:
        f.write('configs')

    ori_cfg_dict = dict()
    ori_cfg_dict['cfg_name'] = osp.basename(temp_file.name)
    ori_cfg_dict['work_dir'] = 'work_dirs/${cfg_name}/${percent}/${fold}'
    ori_cfg_dict['percent'] = 5
    ori_cfg_dict['fold'] = 1
    ori_cfg_dict['model_wrapper'] = dict(
        type='SoftTeacher', detector='${model}')
    ori_cfg_dict['model'] = dict(
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
        rpn_proposal_nms='${model.train_cfg.rpn_proposal.nms.iou_threshold}',
        test_rpn_nms='${model.test_cfg.rpn.nms.iou_threshold}',
        test_rcnn_nms='${model.test_cfg.rcnn.nms.iou_threshold}',
    )

    ori_cfg_dict['str'] = 'Hello, world!'
    ori_cfg_dict['dict'] = {'Hello': 'world!'}
    ori_cfg_dict['list'] = [
        'Hello, world!',
    ]
    ori_cfg_dict['tuple'] = ('Hello, world!', )
    ori_cfg_dict['test_str'] = 'xxx${str}xxx'

    ori_cfg = Config(ori_cfg_dict, filename=cfg_path)
    updated_cfg = replace_cfg_vals(deepcopy(ori_cfg))

    assert updated_cfg.work_dir \
        == f'work_dirs/{osp.basename(temp_file.name)}/5/1'
    assert updated_cfg.model.detector == ori_cfg.model
    assert updated_cfg.iou_threshold.rpn_proposal_nms \
        == ori_cfg.model.train_cfg.rpn_proposal.nms.iou_threshold
    assert updated_cfg.test_str == 'xxxHello, world!xxx'
    ori_cfg_dict['test_dict'] = 'xxx${dict}xxx'
    ori_cfg_dict['test_list'] = 'xxx${list}xxx'
    ori_cfg_dict['test_tuple'] = 'xxx${tuple}xxx'
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_dict'] = 'xxx${dict}xxx'
        updated_cfg = replace_cfg_vals(cfg)
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_list'] = 'xxx${list}xxx'
        updated_cfg = replace_cfg_vals(cfg)
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_tuple'] = 'xxx${tuple}xxx'
        updated_cfg = replace_cfg_vals(cfg)
