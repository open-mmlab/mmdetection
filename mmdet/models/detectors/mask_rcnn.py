from .two_stage import TwoStageDetector


class MaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        # TODO: show segmentation masks
        assert isinstance(result, tuple)
        assert len(result) == 2  # (bbox_results, segm_results)
        super(MaskRCNN, self).show_result(data, result[0], img_norm_cfg,
                                          **kwargs)
