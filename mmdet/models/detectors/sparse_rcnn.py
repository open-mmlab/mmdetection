from torch import nn

from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class SparseRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(
        self,
        *args,
        num_proposals=100,
        hidden_dim=256,
        **kwargs,
    ):
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)
        # there is init_weight in super().__init__,
        # so we can't move this init_weight to self.init_weight

        # TODO move this to bbox_head may be better ?
        # or make it as rpn_head to be consistent with
        #  two stage
        self.init_proposal_features = nn.Embedding(self.num_proposals,
                                                   self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None
        assert gt_masks is None
        h, w = img.size()[-2:]
        num_imgs = len(img_metas)
        imgs_whwh = img.new_tensor([w, h, w, h])[None].expand(img.size(0), 4)
        init_proposal_boxes = self.init_proposal_boxes.weight.clone()
        init_proposal_boxes = bbox_cxcywh_to_xyxy(init_proposal_boxes)
        init_proposal_boxes = init_proposal_boxes[None] * imgs_whwh[:, None, :]

        init_proposal_features = self.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].repeat(
            1, num_imgs, 1)
        x = self.extract_feat(img)
        losses = dict()
        roi_losses = self.roi_head.forward_train(
            x,
            init_proposal_boxes,
            init_proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            # all with same whwh in single batch
            imgs_whwh=imgs_whwh[0])
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals is None
        x = self.extract_feat(img)
        h, w = img.size()[-2:]
        num_imgs = len(img_metas)
        imgs_whwh = img.new_tensor([w, h, w, h])[None].expand(img.size(0), 4)
        init_proposal_boxes = self.init_proposal_boxes.weight.clone()
        init_proposal_boxes = bbox_cxcywh_to_xyxy(init_proposal_boxes)
        init_proposal_boxes = init_proposal_boxes[None] * imgs_whwh[:, None, :]
        init_proposal_features = self.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].repeat(
            num_imgs, 1, 1)
        return self.roi_head.simple_test(
            x,
            init_proposal_boxes,
            init_proposal_features,
            img_metas,
            rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        raise NotImplementedError
