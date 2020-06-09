from mmdet.core import merge_aug_proposals
from .base_dense_head import BaseDenseHead


class BaseRPNHead(BaseDenseHead):
    """Base class for rpn heads"""

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None):
        outs = self.__call__(x)
        loss_inputs = outs + (gt_bboxes, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def aug_test(self, feats, img_metas):
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals
