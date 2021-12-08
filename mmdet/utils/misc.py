# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp


def find_latest_checkpoint(path, ext='pth'):
    """
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f'latest.{ext}')):
        return osp.join(path, f'latest.{ext}')

    checkpoints = glob.glob(osp.join(path, f'*.{ext}'))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path
