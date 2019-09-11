import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from mmdet.core import bbox_overlaps, delta2bbox, bbox2delta
from .retina_head import RetinaHead
from ..registry import HEADS


class Clip(Function):

    @staticmethod
    def forward(ctx, x, a, b):
        return x.clamp(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output, None, None


clip = Clip.apply


@HEADS.register_module
class FreeAnchorRetinaHead(RetinaHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(FreeAnchorRetinaHead,
              self).__init__(num_classes, in_channels, stacked_convs,
                             octave_base_scale, scales_per_octave, conv_cfg,
                             norm_cfg, **kwargs)

        self.iou_threshold = 0.3
        self.pre_anchor_topk = 50
        self.smooth_l1_loss_param = (0.75, 0.11)
        self.bbox_threshold = 0.6
        self.focal_loss_alpha = 0.5
        self.focal_loss_gamma = 2.0

        self.positive_bag_loss_func = positive_bag_loss
        self.negative_bag_loss_func = focal_loss

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        anchors = [torch.cat(anchor) for anchor in anchor_list]

        box_cls_flattened = []
        box_regression_flattened = []
        for box_cls_per_level, box_regression_per_level in zip(
                cls_scores, bbox_preds):
            N, A, H, W = box_cls_per_level.shape
            C = self.cls_out_channels
            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)
            box_regression_per_level = box_regression_per_level.view(
                N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(
                0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(
                N, -1, 4)
            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)

        box_cls = torch.cat(box_cls_flattened, dim=1)
        box_regression = torch.cat(box_regression_flattened, dim=1)

        cls_prob = torch.sigmoid(box_cls)
        box_prob = []
        positive_numels = 0
        positive_losses = []
        for img, (anchors_, labels_, targets_, cls_prob_,
                  box_regression_) in enumerate(
                      zip(anchors, gt_labels, gt_bboxes, cls_prob,
                          box_regression)):
            labels_ -= 1

            with torch.set_grad_enabled(False):
                box_localization = delta2bbox(anchors_, box_regression_,
                                              self.target_means,
                                              self.target_stds)
                object_box_iou = bbox_overlaps(targets_, box_localization)
                H = object_box_iou.max(
                    dim=1,
                    keepdim=True).values.clamp(min=self.bbox_threshold + 1e-12)
                object_box_prob = ((object_box_iou - self.bbox_threshold) /
                                   (H - self.bbox_threshold)).clamp(
                                       min=0, max=1)

                indices = torch.stack(
                    [torch.arange(len(labels_)).type_as(labels_), labels_],
                    dim=0)

                indices = torch.nonzero(
                    torch.sparse.sum(
                        torch.sparse_coo_tensor(indices, object_box_prob),
                        dim=0).to_dense()).t_()

                if indices.numel() == 0:
                    image_box_prob = torch.zeros(
                        anchors_.size(0),
                        self.cls_out_channels).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (labels_.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor(
                            [0]).type_as(object_box_prob)).max(dim=0).values

                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]),
                        nonzero_box_prob,
                        size=(anchors_.size(0),
                              self.cls_out_channels)).to_dense()

                box_prob.append(image_box_prob)

            match_quality_matrix = bbox_overlaps(targets_, anchors_)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                              1)).squeeze(2)

            matched_anchors = anchors_[matched]
            matched_object_targets = bbox2delta(
                matched_anchors,
                targets_.unsqueeze(dim=1).expand_as(matched_anchors),
                self.target_means, self.target_stds)
            retinanet_regression_loss = smooth_l1_loss(
                box_regression_[matched], matched_object_targets,
                *self.smooth_l1_loss_param)
            matched_box_prob = torch.exp(-retinanet_regression_loss)

            positive_numels += len(targets_)
            positive_losses.append(
                self.positive_bag_loss_func(
                    matched_cls_prob * matched_box_prob, dim=1))
        positive_loss = torch.cat(positive_losses).sum() / max(
            1, positive_numels)

        box_prob = torch.stack(box_prob, dim=0)

        negative_loss = self.negative_bag_loss_func(
            cls_prob * (1 - box_prob), self.focal_loss_gamma) / max(
                1, positive_numels * self.pre_anchor_topk)

        losses = {
            "loss_retina_positive": positive_loss * self.focal_loss_alpha,
            "loss_retina_negative":
            negative_loss * (1 - self.focal_loss_alpha),
        }
        return losses


def smooth_l1_loss(pred, target, weight, beta):
    val = target - pred
    abs_val = val.abs()
    smooth_mask = abs_val < beta
    return weight * torch.where(smooth_mask, 0.5 / beta * val**2,
                                (abs_val - 0.5 * beta)).sum(dim=-1)


def positive_bag_loss(logits, *args, **kwargs):
    weight = 1 / clip(1 - logits, 1e-12, None)
    weight /= weight.sum(*args, **kwargs).unsqueeze(dim=-1)
    bag_prob = (weight * logits).sum(*args, **kwargs)
    return F.binary_cross_entropy(
        bag_prob, torch.ones_like(bag_prob), reduction='none')


def focal_loss(logits, gamma):
    return torch.sum(logits**gamma * F.binary_cross_entropy(
        logits, torch.zeros_like(logits), reduction='none'))
