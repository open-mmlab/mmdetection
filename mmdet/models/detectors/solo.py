import torch

from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class SOLO(SingleStageInstanceSegmentor):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(SOLO, self).__init__(
            backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

    def onnx_export(self, img, img_metas):

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        feat = self.extract_feat(img)
        masks = self.mask_head.onnx_export(feat, img_metas)
        return masks
