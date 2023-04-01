# Copyright (c) OpenMMLab. All rights reserved.
import torch


def split_batch(img, img_metas, kwargs):
    """Split data_batch by tags.

    Code is modified from
    <https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/structure_utils.py> # noqa: E501

    Args:
        img (Tensor): of shape (N, C, H, W) encoding input images.
            Typically these should be mean centered and std scaled.
        img_metas (list[dict]): List of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.
        kwargs (dict): Specific to concrete implementation.

    Returns:
        data_groups (dict): a dict that data_batch splited by tags,
            such as 'sup', 'unsup_teacher', and 'unsup_student'.
    """

    # only stack img in the batch
    def fuse_list(obj_list, obj):
        return torch.stack(obj_list) if isinstance(obj,
                                                   torch.Tensor) else obj_list

    # select data with tag from data_batch
    def select_group(data_batch, current_tag):
        group_flag = [tag == current_tag for tag in data_batch['tag']]
        return {
            k: fuse_list([vv for vv, gf in zip(v, group_flag) if gf], v)
            for k, v in data_batch.items()
        }

    kwargs.update({'img': img, 'img_metas': img_metas})
    kwargs.update({'tag': [meta['tag'] for meta in img_metas]})
    tags = list(set(kwargs['tag']))
    data_groups = {tag: select_group(kwargs, tag) for tag in tags}
    for tag, group in data_groups.items():
        group.pop('tag')
    return data_groups
