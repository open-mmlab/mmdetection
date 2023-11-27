import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import nms

from mmdet.structures.bbox import bbox_overlaps
from ..builder import build_queue
from ..queues import Queues
from .utils import SinePositionalEncoding


def box_area(boxes):
    whs = boxes[:, 2:] - boxes[:, :2]
    return whs.prod(dim=-1)


class BaronBase(nn.Module):

    def __init__(self, num_words, word_dim, words_drop_ratio, queue_cfg,
                 sampling_cfg, **kwargs):
        super().__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.word_dropout = words_drop_ratio
        if 'type' in queue_cfg:
            self.queues = build_queue(queue_cfg)
        else:
            self.queues = Queues(**queue_cfg)
        self.sampling_cfg = sampling_cfg
        self.positional_embed = SinePositionalEncoding(
            num_feats=128, num_words=num_words, word_dims=word_dim)

    @staticmethod
    def preprocess_proposals(proposals, image_boxes, shape_ratio_thr,
                             area_ratio_thr, objectness_thr, nms_thr):
        if len(proposals) == 0:
            return proposals
        iofs = bbox_overlaps(
            image_boxes, proposals.bboxes, mode='iof', is_aligned=False)[0]

        size_of_boxes = proposals.bboxes[:, 2:4] - proposals.bboxes[:, :2]
        boxes_shape_ratio = size_of_boxes[:, 0] / (size_of_boxes[:, 1] + 1e-12)

        assert shape_ratio_thr < 1.0

        valid_shape_ratio = torch.logical_and(
            shape_ratio_thr < boxes_shape_ratio, boxes_shape_ratio <
            (1.0 / shape_ratio_thr))
        valid_area = iofs > area_ratio_thr
        valid_object_score = proposals.scores > objectness_thr
        valid_shape = torch.logical_and(valid_shape_ratio, valid_area)

        all_valid = torch.logical_and(valid_shape, valid_object_score)
        if all_valid.sum() < 1:
            all_valid[proposals.scores.argmax()] = True
            return proposals[all_valid]

        proposals = proposals[all_valid]

        _, nms_kept = nms(
            boxes=proposals.bboxes,
            scores=proposals.scores,
            iou_threshold=nms_thr)

        return proposals[nms_kept]

    def _drop_word(self, word_embeddings):
        p = self.word_dropout
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(
            word_embeddings.new_ones(num_preds, num_words),
            p=p,
            training=self.training)
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0
        mask = mask > 0.0

        return mask
