from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule, force_fp32


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_preds(self,
                  cls_scores,
                  bbox_preds,
                  score_factors=None,
                  img_metas=None,
                  rescale=False,
                  with_nms=True,
                  cfg=None,
                  **kwargs):
        """Transform network output for a batch into bbox predictions."""
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is not None:
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if score_factors is not None:
                score_factor_list = [
                    score_factors[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self.get_preds_single(cls_score_list, bbox_pred_list,
                                            score_factor_list, img_meta, cfg,
                                            rescale, with_nms, **kwargs)
            result_list.append(results)
        return result_list

    def get_preds_single(self,
                         cls_score_list,
                         bbox_pred_list,
                         score_factor_list,
                         img_meta,
                         cfg=None,
                         rescale=False,
                         with_nms=True,
                         **kwargs):
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)
