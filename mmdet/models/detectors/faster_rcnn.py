from .two_stage import TwoStageDetector


class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
                    backbone=backbone,
                    neck=neck,
                    rpn_head=rpn_head,
                    bbox_roi_extractor=bbox_roi_extractor,
                    bbox_head=bbox_head,
                    train_cfg=train_cfg,
                    test_cfg=test_cfg,
                    pretrained=pretrained)
