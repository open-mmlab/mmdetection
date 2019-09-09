from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FastRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 pretrained=None):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            pretrained=pretrained)

    def forward_test(self, imgs, img_metas, proposals, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], proposals[0],
                                    **kwargs)
        else:
            return self.aug_test(imgs, img_metas, proposals, **kwargs)
