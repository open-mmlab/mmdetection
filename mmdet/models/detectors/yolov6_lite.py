# Migrate based on Meituan yolov6 lite

from mmdet.registry import MODELS
from .single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

@MODELS.register_module()
class YOLOV6Lite(SingleStageDetector):
    """
    commit here
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)