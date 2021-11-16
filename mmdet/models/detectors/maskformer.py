from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class Maskformer(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is 
    NOT All You Need for Semantic Segmentation 
    <https://arxiv.org/pdf/2107.06278>`"""

    def __init__(self, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, # not used
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None,
                 # for panoptic segmentation
                 semantic_head=None):
        super().__init__(backbone, neck=neck, bbox_head=bbox_head, train_cfg=train_cfg, 
                         test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg)
        
        assert semantic_head is not None
        self.semantic_head = build_head(semantic_head)

    @property
    def with_semantic_head(self):
        return (hasattr(self, 'semantic_head') and self.semantic_head is not None)


    def forward_train(self, 
                      img, 
                      img_metas, 
                      gt_bboxes, 
                      gt_labels, 
                      gt_masks, 
                      gt_semantic_seg,
                      gt_bboxes_ignore=None,
                      **kargs):
        """
        Args:
            img (Tensor): shape = (B, C, H, W).
            img_metas (list[Dict]): image metas.
            gt_bboxes (list[Tensor]): shape of each element = (num_objects, 4).
            gt_labels (list[Tensor]): shape of each element = (num_objects, ).
            gt_masks (list[BitmapMasks]): 表示不同的实例的mask.
            gt_semantic_seg (list[tensor]): thing 在前 0-79， stuff在后 80-132, 255为背景. 
            gt_bboxes_ignore (list[Tensor]): # TODO 不知道是否需要考虑掉这些实例. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = self.sem_seg_head.forward_train(x, img_metas, 
                                                        gt_bboxes, gt_labels, gt_bboxes_ignore, 
                                                        gt_masks, gt_semantic_seg)
        
        return losses

    
    def simple_test(self, img, img_metas, **kwargs):
        feat = self.extract_feat(img)
        mask_results = self.semantic_head.simple_test(
            feat, img_metas, **kwargs)

        return mask_results


    def show_result(self):
        # TODO show the panoptic result
        raise NotImplementedError


    def onnx_export(self):
        # TODO
        raise NotImplementedError