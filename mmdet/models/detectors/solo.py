from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class SOLO(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(SOLO, self).__init__(backbone=backbone,
                                     neck=neck,
                                     bbox_head=bbox_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     pretrained=pretrained,
                                     init_cfg=init_cfg)
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, gt_masks)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_results, segm_results = self.bbox_head.get_seg(*seg_inputs)
        return [(bbox_results[0], segm_results[0])]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
