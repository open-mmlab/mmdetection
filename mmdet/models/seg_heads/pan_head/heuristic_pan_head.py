from abc import ABCMeta

import torch
import torch.nn as nn

from ...builder import build_loss


class BasePanHead(nn.Module, metaclass=ABCMeta):

    def __init__(
        self,
        num_things=80,
        num_stuff=53,
        loss_panoptic=None,
    ):
        self.num_things = num_things
        self.num_stuff = num_stuff

        if loss_panoptic:
            self.loss_panoptic = build_loss(loss_panoptic)
        else:
            self.loss_panoptic = None

    @property
    def with_loss(self):
        return self.loss_panoptic is not None

    def loss(self):
        pass

    def lay_masks(bboxes, labels, segm_masks, img_meta, overlap_thr=0.5):
        img_h, img_w, _ = img_meta[0]['ori_shape']
        num_insts = bboxes.shape[0]
        id_map = torch.zeros([img_h, img_w],
                             device=labels.device,
                             dtype=torch.long)
        if num_insts == 0:
            return id_map, labels

        scores, bboxes = bboxes[:, -1], bboxes[:, :4]

        # for unmatched shits, order by scores
        order = torch.argsort(-scores)
        bboxes = bboxes[order]
        labels = labels[order]
        segm_masks = segm_masks[order]

        inst_idx = 1
        left_labels = []
        for idx in range(bboxes.shape[0]):
            _cls = labels[idx]
            _mask = segm_masks[idx]
            inst_id_map = torch.ones_like(_mask, dtype=torch.long) * inst_idx
            area = _mask.sum()
            if area == 0:
                continue

            used = id_map > 0
            intersect = (_mask * used).sum()
            if (intersect / (area + 1e-5)) > overlap_thr:
                continue

            _part = _mask * (~used)
            id_map = torch.where(_part, inst_id_map, id_map)
            left_labels.append(_cls)
            inst_idx += 1

        inst_labels = torch.stack(left_labels)
        assert inst_idx == (len(inst_labels) + 1)
        return id_map, inst_labels

    def simple_test(
        self,
        det_bboxes,
        det_labels,
        mask_preds,
        seg_logits,
    ):
        pass
