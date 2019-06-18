import torch


def reduce_vision(x, mask_size):
    quater_size = mask_size // 4
    base = ((0, quater_size * 2), (quater_size, quater_size * 3),
            (quater_size * 2, quater_size * 4))
    layers = [
        x[:, i:i + 1][:, :, base[i % 3][0]:base[i % 3][1],
                      base[i // 3][0]:base[i // 3][1]] for i in range(9)]
    layers = torch.cat(layers, dim=1)
    return layers


def grid_target(sampling_results, cfg):
    # Mix all samples(across images) together.
    pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results], dim=0)
    pos_gt_bboxes = torch.cat([res.pos_gt_bboxes for res in sampling_results],
                              dim=0)
    assert (pos_bboxes.shape == pos_gt_bboxes.shape)

    # Expand pos_bboxes
    x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
    y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
    x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
    y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2

    pos_bboxes = torch.cat(
        list(map(lambda x: x.view(-1, 1), [x1, y1, x2, y2])), dim=1)

    R = pos_bboxes.shape[0]
    G = cfg.num_grids
    mask_size = cfg.mask_size
    targets = torch.zeros([R, G, mask_size, mask_size])

    for rix in range(R):
        for gix in range(G):
            gridpoint_x = (1 - gix // 3 / 2) * (pos_gt_bboxes[rix, 0]) + (
                gix // 3 / 2) * (pos_gt_bboxes[rix, 2])
            gridpoint_y = (1 - gix % 3 / 2) * (pos_gt_bboxes[rix, 1]) + (
                gix % 3 / 2) * (pos_gt_bboxes[rix, 3])
            if (pos_bboxes[rix, 2] - pos_bboxes[rix, 0]) < 4 or (
                    pos_bboxes[rix, 3] - pos_bboxes[rix, 1]) < 4:
                continue
            cx = int((gridpoint_x - pos_bboxes[rix, 0]) /
                     (pos_bboxes[rix, 2] - pos_bboxes[rix, 0]) * mask_size)
            cy = int((gridpoint_y - pos_bboxes[rix, 1]) /
                     (pos_bboxes[rix, 3] - pos_bboxes[rix, 1]) * mask_size)
            radius = cfg.get('radius', 1)
            for x in range(cx - radius, cx + radius + 1):
                for y in range(cy - radius, cy + radius + 1):
                    if x >= 0 and x < mask_size and y >= 0 and y < mask_size:
                        if (x - cx)**2 + (y - cy)**2 <= radius**2:
                            targets[rix, gix, y, x] = 1
    targets = reduce_vision(targets, mask_size)
    targets = targets.float().cuda()
    return targets
