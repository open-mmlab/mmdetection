# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio
from mmseg.registry import DATASETS

from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class iSAIDDataset(BaseSegDataset):
    """ iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    In segmentation map annotation for iSAID dataset, which is included
    in 16 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    METAINFO = dict(
        classes=('background', 'ship', 'store_tank', 'baseball_diamond',
                 'tennis_court', 'basketball_court', 'Ground_Track_Field',
                 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                 'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
                 'Harbor'),
        palette=[[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
                 [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
                 [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
                 [0, 127, 191], [0, 127, 255], [0, 100, 155]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_instance_color_RGB.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
