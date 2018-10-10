from .two_stage import TwoStageDetector


class FastRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 mask_roi_extractor=None,
                 mask_head=None,
                 pretrained=None):
        super(FastRCNN, self).__init__(
                    backbone=backbone,
                    neck=neck,
                    bbox_roi_extractor=bbox_roi_extractor,
                    bbox_head=bbox_head,
                    train_cfg=train_cfg,
                    test_cfg=test_cfg,
                    mask_roi_extractor=mask_roi_extractor,
                    mask_head=mask_head,
                    pretrained=pretrained)
