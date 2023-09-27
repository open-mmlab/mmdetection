from mmengine.config import read_base
from projects.example_largemodel import name_auto_wrap_policy

with read_base():
   from mmdet.configs.dino.dino_5scale_swin_l_8xb2_12e_coco import *

custom_imports = dict(
    imports=['projects.example_largemodel'], allow_failed_imports=False)

runner_type = 'FlexibleRunner'

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type=name_auto_wrap_policy)))
