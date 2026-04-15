# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class SingleTreeCocoDataset(CocoDataset):
    """Demo dataset for single-tree instance segmentation in COCO format.

    This dataset is intended for a binary segmentation setup where:

    - background is implicit
    - the only foreground category is ``single_tree``

    The underlying annotation format is standard COCO detection/instance
    segmentation format. Image size is usually handled by the config pipeline,
    but this demo class optionally checks for fixed ``1024x1024`` inputs.

    Args:
        expected_size (Tuple[int, int]): Expected image size as
            ``(width, height)``. Defaults to ``(1024, 1024)``.
        enforce_size (bool): Whether to raise an error if an image does not
            match ``expected_size``. Defaults to ``False``.
    """

    METAINFO = {
        'classes': ('single_tree', ),
        'palette': [(34, 139, 34)],
    }

    def __init__(self,
                 *args,
                 expected_size: Tuple[int, int] = (1024, 1024),
                 enforce_size: bool = False,
                 **kwargs) -> None:
        self.expected_size = expected_size
        self.enforce_size = enforce_size
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse one COCO-style sample and optionally validate image size."""
        data_info = super().parse_data_info(raw_data_info)

        if self.enforce_size:
            expected_w, expected_h = self.expected_size
            if (data_info['width'], data_info['height']) != (
                    expected_w, expected_h):
                raise ValueError(
                    'SingleTreeCocoDataset expects images with size '
                    f'{expected_w}x{expected_h}, but got '
                    f"{data_info['width']}x{data_info['height']} for "
                    f"img_id={data_info['img_id']}.")

        return data_info
