# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcywh


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


@TASK_UTILS.register_module()
class BBoxL1Cost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # convert box format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class IoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode: str = 'giou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ...  match_costs.match_cost import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = pred_scores.softmax(-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight


@TASK_UTILS.register_module()
class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
                in shape (num_queries, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if self.binary_input:
            pred_masks = pred_instances.masks
            gt_masks = gt_instances.masks
            return self._mask_focal_loss_cost(pred_masks, gt_masks)
        else:
            pred_scores = pred_instances.scores
            gt_labels = gt_instances.labels
            return self._focal_loss_cost(pred_scores, gt_labels)


@TASK_UTILS.register_module()
class BinaryFocalLossCost(FocalLossCost):

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # gt_instances.text_token_mask is a repeated tensor of the same length
        # of instances. Only gt_instances.text_token_mask[0] is useful
        text_token_mask = torch.nonzero(
            gt_instances.text_token_mask[0]).squeeze(-1)
        pred_scores = pred_instances.scores[:, text_token_mask]
        gt_labels = gt_instances.positive_maps[:, text_token_mask]
        return self._focal_loss_cost(pred_scores, gt_labels)


@TASK_UTILS.register_module()
class DiceCost(BaseMatchCost):
    """Cost of mask assignments based on dice losses.

    Args:
        pred_act (bool): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float): Defaults to 1e-3.
        naive_dice (bool): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If False, use the second power that
            is adopted by K-Net and SOLO. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 pred_act: bool = False,
                 eps: float = 1e-3,
                 naive_dice: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def _binary_mask_dice_loss(self, mask_preds: Tensor,
                               gt_masks: Tensor) -> Tensor:
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_queries, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (num_queries, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                          gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                          gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks

        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


@TASK_UTILS.register_module()
class CrossEntropyLossCost(BaseMatchCost):
    """CrossEntropyLossCost.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 use_sigmoid: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred: Tensor,
                              gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_queries, 1, *) or
                (num_queries, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``masks``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError

        return cls_cost * self.weight
