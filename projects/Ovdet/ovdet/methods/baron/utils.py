import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normed_boxes(boxes, spanned_box):
    boxes = boxes[:, :4]
    spanned_box_shape = spanned_box[2:] - spanned_box[:2]
    boxes = boxes.view(-1, 2, 2) - spanned_box[:2].view(1, 1, 2)
    boxes = boxes / (spanned_box_shape.view(1, 1, 2) + 1e-12)

    return boxes.view(-1, 4)


# repeat crops and get attention masks
def repeat_crops_and_get_att_mask(crops,
                                  repeat_nums,
                                  normed_boxes,
                                  num_heads,
                                  grid_size=7,
                                  use_attn_mask=True):
    repeated_crops = torch.cat([
        crop[None].repeat(repeat_num, 1, 1, 1)
        for crop, repeat_num in zip(crops, repeat_nums)
    ],
                               dim=0)
    if use_attn_mask:
        boxes_split_by_seqs = [n.shape[0] for n in normed_boxes]
        normed_boxes = torch.cat(normed_boxes)
        masks_per_box = get_att_mask_by_matrix(normed_boxes, grid_size)
        masks_split_by_seqs = masks_per_box.split(boxes_split_by_seqs, dim=0)
        masks_split_by_seqs = [ms.sum(0) for ms in masks_split_by_seqs]
        masks_split_by_seqs = torch.stack(masks_split_by_seqs, dim=0)
        mask_flatten = masks_split_by_seqs.flatten(-2, -1)
        mask_flatten = torch.cat(
            [torch.ones_like(mask_flatten[..., :1]), mask_flatten], dim=-1)
        attn_mask = mask_flatten[..., None] * mask_flatten[:, None, :]
        attn_mask = torch.where(attn_mask > 0.0, 0.0, float('-inf'))
        attn_mask[:, range(grid_size**2 + 1), range(grid_size**2 + 1)] = 0.0
        attn_mask = attn_mask[:, None].repeat(1, num_heads, 1, 1)
        attn_mask = attn_mask.flatten(0, 1)
    else:
        attn_mask = None

    return repeated_crops, attn_mask


def get_att_mask_by_matrix(normed_boxes, grid_size):
    boxes = normed_boxes * (grid_size - 1) + 0.5
    boxes = boxes.view(-1, 2, 2)
    num_boxes = boxes.shape[0]
    boxes[:, 0] = boxes[:, 0].floor()
    boxes[:, 1] = boxes[:, 1].ceil()
    boxes = boxes.long()
    x_range_pairs = boxes[..., 0]
    y_range_pairs = boxes[..., 1]
    x_mask, y_mask = single_direction_mask(
        torch.cat([x_range_pairs, y_range_pairs], dim=0), grid_size).split(
            [num_boxes] * 2, dim=0)
    mask = torch.logical_and(
        y_mask.view(num_boxes, grid_size, 1),
        x_mask.view(num_boxes, 1, grid_size))

    return mask


def single_direction_mask(range_pairs, grid_size):
    num_pairs = range_pairs.shape[0]
    device = range_pairs.device
    ref_matrix = torch.arange(grid_size).view(1, -1).repeat(num_pairs,
                                                            1).to(device)
    beg = range_pairs[:, 0:1].repeat(1, grid_size)
    end = range_pairs[:, 1:2].repeat(1, grid_size)
    mask = ref_matrix.ge(beg) & ref_matrix.lt(end)

    return mask


def bboxes_area(bboxes):
    whs = torch.clamp(bboxes[:, 2:4] - bboxes[:, :2], min=0.0)
    return whs.prod(-1)


def soft_ce_loss(x, target, dim=-1):
    loss = torch.sum(
        -target * F.log_softmax(x, dim=dim), dim=dim) / (
            torch.sum(target, dim=dim) + 1e-6)
    return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, dim=-1):
        loss = torch.sum(
            -target * F.log_softmax(x, dim=dim), dim=dim) / (
                torch.sum(target, dim=dim) + 1e-6)
        return loss.mean()


class SinePositionalEncoding(nn.Module):

    def __init__(self,
                 num_feats=128,
                 num_words=4,
                 word_dims=512,
                 temperature=1.2,
                 scale=2 * math.pi):
        super(SinePositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.scale = scale
        self.pos_proj = nn.Sequential(
            nn.Linear(num_feats * 4, word_dims), nn.LayerNorm(word_dims),
            nn.Linear(word_dims, num_words * word_dims))
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):
        embed = x * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(
            dim_t.div(2, rounding_mode='floor') - (self.num_feats // 4))
        pos = embed[:, :, None] * dim_t[None, None]
        pos[..., 0::2] = pos[..., 0::2].sin()
        pos[..., 1::2] = pos[..., 1::2].cos()

        assert pos.shape[-1] == self.num_feats

        pos = pos.view(-1, 4 * self.num_feats)

        return self.pos_proj(pos).view(-1, self.num_words, self.word_dims)
