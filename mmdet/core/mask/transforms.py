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

import mmcv
import numpy as np
import pycocotools.mask as mask_util

from ..utils.misc import to_numpy


def mask2result(det_bboxes,
                det_labels,
                det_masks,
                num_classes,
                mask_thr_binary=0.5,
                rle=True,
                full_size=True,
                img_size=None):

    masks = to_numpy(det_masks)
    bboxes = to_numpy(det_bboxes, np.int32)[:, :4]
    labels = to_numpy(det_labels, np.int32)

    cls_masks = [[] for _ in range(num_classes - 1)]

    for bbox, label, mask in zip(bboxes, labels, masks):
        w = max(bbox[2] - bbox[0] + 1, 1)
        h = max(bbox[3] - bbox[1] + 1, 1)

        mask = mmcv.imresize(mask, (w, h))
        mask = (mask > mask_thr_binary).astype(np.uint8)

        if full_size:
            assert img_size is not None
            im_mask = np.zeros(img_size[:2], dtype=np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = mask
            mask = im_mask

        if rle:
            mask = mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F'))[0]

        cls_masks[label].append(mask)

    return cls_masks
