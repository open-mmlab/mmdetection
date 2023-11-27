import math
import random

import numpy as np
import torch

from ...utils.misc import multi_apply


def get_normed_boxes(boxes, spanned_box):
    spanned_box_shape = spanned_box[2:] - spanned_box[:2]
    boxes = boxes.reshape(-1, 2, 2) - spanned_box[:2].reshape(1, 1, 2)
    boxes = boxes / (spanned_box_shape.reshape(1, 1, 2) + 1e-12)

    return boxes.reshape(-1, 4)


def get_spanned_box(boxes, image_size=None):
    # boxes x, y
    # image_size h, w
    corner_points = boxes.reshape(-1, 2)
    bottom_right = corner_points.max(0)
    upper_left = corner_points.min(0)

    upper_left_bottom_right = np.stack([upper_left, bottom_right], axis=0)
    if image_size is not None:
        upper_left_bottom_right = clamp_with_image_size(
            upper_left_bottom_right, image_size)

    return upper_left_bottom_right.reshape(4)


def clamp_with_image_size(points, image_size):
    points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=image_size[1])
    points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=image_size[0])

    return points


def perm_generator(seq):
    seen = set()
    length = len(seq)
    while True:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            yield perm


def pseudo_permutations(seq_length, num_permutation):
    rand_perms = perm_generator(list(range(seq_length)))
    return [list(next(rand_perms)) for _ in range(num_permutation)]


class NeighborhoodSampling:
    """
        checkboard:   0  1  2
                      3  4  5
                      6  7  8
        context boxes: [0, 1, 2, 3, 5, 6, 7, 8]
        candidate_groups: 2 ** 8 = 256
        box: tensor
    """

    def __init__(self,
                 max_groups=4,
                 max_permutations=4,
                 alpha=3.0,
                 cut_off_thr=0.5,
                 base_probability=0.5,
                 interval=0.0,
                 **kwargs):
        self.interval = interval
        box_ids = []
        left_right_up_downs = []
        box_templates = []
        for i in range(3):
            h_interval = (float(i) - 1.0) * self.interval
            for j in range(3):
                w_interval = (float(j) - 1.0) * self.interval
                box = [
                    float(j) + w_interval,
                    float(i) + h_interval,
                    float(j + 1) + w_interval,
                    float(i + 1) + h_interval
                ]
                box_templates.append(box)
        self.box_templates = np.array(box_templates, dtype=np.float32)
        self.binary_mask_template = 10**np.arange(9, dtype=np.float32)

        for left in range(2):  # left: -1
            for r in range(2):  # right +1
                for u in range(2):  # up -3
                    for d in range(2):  # down  +3
                        left_right_up_downs.append([left, r, u, d])
                        box_ids.append(
                            list({
                                4 - left - 3 * u, 4 - 3 * u, 4 + r - 3 * u,
                                4 - left, 4, 4 + r, 4 - left + 3 * d,
                                4 + 3 * d, 4 + r + 3 * d
                            }))
        self.box_ids = box_ids
        self.alpha = alpha
        self.cut_off_thr = cut_off_thr
        self.left_right_up_downs = np.array(
            left_right_up_downs, dtype=np.float32)
        self.max_groups = max_groups
        self.max_permutations = max_permutations
        self.base_probability = base_probability
        self.context_box_ids = [0, 1, 2, 3, 5, 6, 7, 8]

    @staticmethod
    def _get_group_id(left_right_up_down):
        assert len(left_right_up_down) == 4
        # list of {0, 1}'s
        left, right, up, down = left_right_up_down
        return left * (2**3) + right * (2**2) + up * 2 + down

    def _get_left_right_up_down_possibility(self, box, image_size):
        img_h, img_w = image_size
        box_w, box_h = box[2] - box[0] + 1e-12, box[3] - box[1] + 1e-12
        box_h_w_ratio = box_h / box_w
        box_w_h_ratio = box_w / box_h
        # Initiate: <, >, ^, v
        left_right_up_down = (np.array(
            [box_h_w_ratio, box_h_w_ratio, box_w_h_ratio, box_w_h_ratio],
            dtype=np.float32)**self.alpha) * self.base_probability
        # check boundary
        boundary_check = np.array([
            box[0] / box_w, (img_w - box[2]) / box_w, box[1] / box_h,
            (img_h - box[3]) / box_h
        ],
                                  dtype=np.float32) > (
                                      self.cut_off_thr + self.interval)
        left_right_up_down = left_right_up_down * boundary_check.astype(
            np.float32)
        left_right_up_down = np.clip(
            left_right_up_down, a_min=0.0, a_max=self.base_probability)
        left_right_up_down[np.isnan(left_right_up_down)] = 0.0

        return left_right_up_down

    def group_generator(self, box_possibilities):
        assert box_possibilities[4] == 1.0  # center box (roi) are fixed at 1.0
        seen = set()
        while True:
            try:
                sampled_mask = torch.bernoulli(
                    torch.from_numpy(box_possibilities)).numpy()
            except TypeError:
                raise ValueError(
                    f'Invalid box_possibilities{box_possibilities}')
            box_ids = sorted(sampled_mask.nonzero()[0].tolist())
            box_ids_str = ''.join([str(box_id) for box_id in box_ids])
            if box_ids_str not in seen:
                seen.add(box_ids_str)
                yield box_ids

    @staticmethod
    def _get_box_possibilities(left_right_up_down_possibility):
        box_possibilities = np.ones(9, dtype=np.float32)
        box_possibilities[[0, 3, 6]] *= left_right_up_down_possibility[0]
        box_possibilities[[2, 5, 8]] *= left_right_up_down_possibility[1]
        box_possibilities[[0, 1, 2]] *= left_right_up_down_possibility[2]
        box_possibilities[[6, 7, 8]] *= left_right_up_down_possibility[3]
        box_possibilities[[0, 2, 6, 8]] **= 0.5

        return box_possibilities

    def sample(self, box, image_size):
        left_right_up_down_possibility = \
            self._get_left_right_up_down_possibility(box, image_size)
        box_possibilities = self._get_box_possibilities(
            left_right_up_down_possibility)
        box_possibilities = np.clip(box_possibilities, a_min=0.0, a_max=1.0)
        num_valid_context_boxes = int((box_possibilities > 0.0).sum()) - 1
        num_groups = min(self.max_groups,
                         math.factorial(num_valid_context_boxes))
        random_gen = self.group_generator(box_possibilities)

        box_ids_per_group = [next(random_gen) for _ in range(num_groups)]

        box_w, box_h = box[2] - box[0] + 1e-12, box[3] - box[1] + 1e-12
        box_templates = self.box_templates
        box_templates = box_templates * np.array(
            [box_w, box_h, box_w, box_h], dtype=np.float32).reshape(1, 4)
        center_box_template = box_templates[4]  # [1, 1, 2, 2]
        off_set = np.array(box, dtype=np.float32) - center_box_template
        box_templates = box_templates + off_set.reshape(1, 4)

        groups, normed_boxes, spanned_boxes, box_ids = multi_apply(
            self._sample_boxes_per_group,
            box_ids_per_group,
            image_size=image_size,
            box_templates=box_templates)
        box_ids = [box_id for box_ids_ in box_ids for box_id in box_ids_]

        return groups, normed_boxes, spanned_boxes, box_ids

    def _sample_boxes_per_group(self, box_ids, image_size,
                                box_templates):  # TODO: parallel

        num_boxes = len(box_ids)

        # pseudo_perm = list(range(num_boxes))
        # all_permutations = [pseudo_perm, pseudo_perm[::-1]]
        boxes = box_templates[box_ids]
        boxes = clamp_with_image_size(boxes.reshape(-1, 2),
                                      image_size).reshape(-1, 4)
        spanned_box = get_spanned_box(boxes)
        normed_boxes = get_normed_boxes(boxes, spanned_box)

        num_permutations = min(
            math.factorial(num_boxes), self.max_permutations)

        box_permutations = pseudo_permutations(num_boxes, num_permutations)

        boxes_list = [
            torch.from_numpy(boxes[perm]) for perm in box_permutations
        ]
        normed_boxes_list = [
            torch.from_numpy(normed_boxes[perm]) for perm in box_permutations
        ]
        box_ids_list = [[box_ids[idx] for idx in perm]
                        for perm in box_permutations]

        return boxes_list, normed_boxes_list, torch.from_numpy(
            spanned_box), box_ids_list


if __name__ == '__main__':
    from time import time

    checkboard_sampling = NeighborhoodSampling(
        max_groups=4, max_permutations=2, alpha=0.0, cut_off_thr=0.5)
    image_sizes = [[1000, 800]] * 1000
    boxes = [[300, 350, 350, 450]] * 1000

    tik = time()
    for image_size, box in zip(image_sizes, boxes):
        groups, normed_boxes, spanned_boxes, box_ids = \
           checkboard_sampling.sample(box, image_size)
    print(time() - tik)
    tik = time()
    groups, normed_boxes, spanned_boxes, box_ids = multi_apply(
        checkboard_sampling.sample, boxes, image_sizes)
    print(time() - tik)
