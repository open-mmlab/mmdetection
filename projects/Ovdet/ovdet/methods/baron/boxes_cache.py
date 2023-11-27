import json

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.ops import nms
from mmengine.structures import InstanceData

from mmdet.structures.bbox import bbox_flip, scale_boxes  # todo consider crop


class BoxesCache(nn.Module):

    def __init__(self,
                 json_path,
                 start_iter=10000,
                 num_proposals=300,
                 nms_thr=0.1,
                 score_thr=0.85,
                 save=False):
        super(BoxesCache, self).__init__()
        with open(json_path, 'r') as f:
            images_info = json.load(f)['images']
        num_images = len(images_info)
        self.image_id2ordered_id = {
            info['id']: ordered_id
            for ordered_id, info in enumerate(images_info)
        }
        boxes = torch.zeros(num_images, num_proposals,
                            5)  # [x1, y1, x2, y2, s]
        self.register_buffer('boxes', boxes, persistent=save)
        self.num_proposals = num_proposals
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.register_buffer(
            'iter', torch.tensor(0, dtype=torch.long), persistent=False)
        self.start_iter = start_iter

    def forward(self, proposals):
        return self.update(proposals)

    @torch.no_grad()
    def update(
        self,
        proposals,
    ):
        self.iter.data += 1
        if self.iter < self.start_iter:
            return proposals
        nms_thr = self.nms_thr
        # TODO: pull cached boxes from all devices
        image_id = proposals.img_id

        bboxes = proposals.bboxes
        scores = proposals.scores

        scale_factor = proposals.scale_factor
        flip = proposals.flip
        flip_direction = proposals.flip_direction
        img_shape = proposals.img_shape

        if image_id not in self.image_id2ordered_id:
            return proposals
        ordered_id = self.image_id2ordered_id[image_id]
        cached_bboxes = self.boxes[ordered_id]
        cached_scores = cached_bboxes[:, -1]

        scaled_cached_bboxes = scale_boxes(cached_bboxes[:, :4], scale_factor)
        flipped_cached_bboxes = bbox_flip(scaled_cached_bboxes, img_shape,
                                          flip_direction) \
            if flip else scaled_cached_bboxes

        merged_boxes = torch.cat([flipped_cached_bboxes, bboxes])
        merged_scores = torch.cat([cached_scores, scores])

        score_kept = merged_scores > self.score_thr
        if score_kept.sum() == 0:
            score_kept = merged_scores.argmax().view(1)

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        _, nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        out = InstanceData(
            bboxes=kept_boxes, scores=kept_scores, metainfo=proposals.metainfo)

        # TODO: transform to the original size
        flipped_bboxes = bbox_flip(bboxes, img_shape, flip_direction) \
            if flip else bboxes
        scaled_back_bboxes = scale_boxes(flipped_bboxes,
                                         [1 / s for s in scale_factor])
        merged_boxes = torch.cat([cached_bboxes[:, :4], scaled_back_bboxes])
        merged_scores = torch.cat([cached_scores, scores])
        score_kept = merged_scores > self.score_thr
        if score_kept.sum() == 0:
            score_kept = merged_scores.argmax().view(1)

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        _, nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        num_update = min(self.num_proposals, len(kept_boxes))
        device = kept_scores.device
        update_cache_to_sync = torch.zeros(
            self.num_proposals, 6, device=device)  # [x,y,x,y,s,order_id]
        update_cache_to_sync[:, -1] = float(ordered_id)  # ordered_id
        update_cache = torch.cat([kept_boxes, kept_scores[:, None]],
                                 dim=1)[:num_update]
        update_cache_to_sync[:num_update, :5] = update_cache

        # sync for updates from other devices
        update_cache = self.sync_multiple_gpus(update_cache_to_sync)
        for update_cache_ in update_cache:
            ordered_id_ = int(update_cache_[0, -1].item())
            self.boxes[ordered_id_] = update_cache_[:, :5].to(device)  # update

        return out

    @staticmethod
    def sync_multiple_gpus(tensor):
        """Performs all_gather operation on the provided tensors.

        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if get_world_size() == 1:
            return [tensor]
        with torch.no_grad():
            tensors_gather = [
                torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(
                tensors_gather, tensor, async_op=False)

        return tensors_gather


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
