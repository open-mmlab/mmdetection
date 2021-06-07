from abc import ABCMeta, abstractmethod

import torch
from mmcv.runner import BaseModule


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
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

    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors,
                    img_metas=None,
                    cfg=None,
                    rescale=False,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            score_factors (list[tensor] | None): The re-weight factors
                for cls_scores, e.g. ``centerness`` of ``fcos_head`` and
                ``iou_preds`` of `paa_head`.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor] or list[tuple[Tensor, Tensor]]:
            When with nms is True, return tuple[Tensor, Tensor] and
            first tensors is det_bboxes of shape [N, num_det, 5] and
            second Tensor is class labels of shape [N, num_det].
            Otherwise, return a list[tuple[Tensor, Tensor]],
            Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.
        """
        if score_factors is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
            len(cls_scores) == len(score_factors)

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        # TODO: @Haian Huang, a unified priori_generator, This func can
        #  be used in both
        #  anchor_free_head and anchor_head
        mlvl_prioris = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i] for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i] for i in range(num_levels)]

        if with_score_factors:
            mlvl_score_factors = [score_factors[i] for i in range(num_levels)]
        else:
            mlvl_score_factors = [None for _ in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_prioris)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_cls_score_factors = []
        for cls_score, bbox_pred, score_factor, prioris in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factors,
                mlvl_prioris):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if with_score_factors:
                score_factor = score_factor.permute(0, 2, 3,
                                                    1).reshape(batch_size,
                                                               -1).sigmoid()
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            prioris = prioris.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                # Mind k<=3480 in TensorRT for TopK
                transformed_inds = scores.shape[1] * batch_inds + topk_inds
                scores = scores.reshape(
                    -1, self.num_classes)[transformed_inds].reshape(
                        batch_size, -1, self.num_classes)
                bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds].reshape(
                    batch_size, -1, 4)
                # prioris has shape (batch, num_prioris, 2) in FCOS, FOVEA etc. # noqa: E501
                # prioris has shape (batch, num_prioris, 4) in Retina, ATSS etc. # noqa: E501
                prioris = prioris.reshape(
                    -1, prioris.size(-1))[transformed_inds].reshape(
                        batch_size, -1, prioris.size(-1))
                if with_score_factors:
                    score_factor = score_factor.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1, 1)

            bboxes = self.bbox_coder.decode(
                prioris, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

            if with_score_factors:
                mlvl_cls_score_factors.append(score_factor)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_cls_score_factors, dim=1)
            batch_mlvl_scores = batch_mlvl_scores * batch_score_factors
        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        else:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
            return det_results
