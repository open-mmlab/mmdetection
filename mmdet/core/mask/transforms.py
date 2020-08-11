# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import numpy as np
import pycocotools.mask as mask_util

from ..utils.misc import to_numpy

import torch
import torch.nn.functional as F


def mask2result(det_bboxes,
                det_labels,
                det_masks,
                num_classes,
                mask_thr_binary=0.5,
                img_size=None):
    masks = det_masks
    bboxes = det_bboxes[:, :4]
    labels = det_labels

    if isinstance(masks, np.ndarray):
        masks = torch.tensor(masks)
        bboxes = torch.tensor(bboxes)
        labels = torch.tensor(labels)

    cls_masks = [[] for _ in range(num_classes)]

    for bbox, label, mask in zip(bboxes, labels, masks):

        mask = mask[None, :, :]
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_size[::-1]

        img_y = torch.arange(
            y0_int, y1_int, device=mask.device, dtype=torch.float32) + 0.5
        img_x = torch.arange(
            x0_int, x1_int, device=mask.device, dtype=torch.float32) + 0.5
        x0, y0, x1, y1 = bbox

        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

        gx = img_x[None, :].expand(img_y.size(0), img_x.size(0))
        gy = img_y[:, None].expand(img_y.size(0), img_x.size(0))
        grid = torch.stack([gx, gy], dim=2)

        img_masks = F.grid_sample(
            mask.to(dtype=torch.float32)[None, :, :, :], grid[None, :, :, :], align_corners=False)

        mask = img_masks[0, 0, :, :]
        mask = (mask >= mask_thr_binary).to(dtype=torch.uint8)

        cls_masks[label].append(to_numpy(mask))

    return cls_masks
