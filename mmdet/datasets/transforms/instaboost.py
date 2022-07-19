# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class InstaBoost(BaseTransform):
    r"""Data augmentation method in `InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting
    <https://arxiv.org/abs/1908.07801>`_.

    Refer to https://github.com/GothicAi/Instaboost for implementation details.


    Required Keys:

    - img (np.uint8)
    - instances

    Modified Keys:

    - img (np.uint8)
    - instances

    Args:
        action_candidate (tuple): Action candidates. "normal", "horizontal", \
            "vertical", "skip" are supported. Defaults to ('normal', \
            'horizontal', 'skip').
        action_prob (tuple): Corresponding action probabilities. Should be \
            the same length as action_candidate. Defaults to (1, 0, 0).
        scale (tuple): (min scale, max scale). Defaults to (0.8, 1.2).
        dx (int): The maximum x-axis shift will be (instance width) / dx.
            Defaults to 15.
        dy (int): The maximum y-axis shift will be (instance height) / dy.
            Defaults to 15.
        theta (tuple): (min rotation degree, max rotation degree). \
            Defaults to (-1, 1).
        color_prob (float): Probability of images for color augmentation.
            Defaults to 0.5.
        hflag (bool): Whether to use heatmap guided. Defaults to False.
        aug_ratio (float): Probability of applying this transformation. \
            Defaults to 0.5.
    """

    def __init__(self,
                 action_candidate: tuple = ('normal', 'horizontal', 'skip'),
                 action_prob: tuple = (1, 0, 0),
                 scale: tuple = (0.8, 1.2),
                 dx: int = 15,
                 dy: int = 15,
                 theta: tuple = (-1, 1),
                 color_prob: float = 0.5,
                 hflag: bool = False,
                 aug_ratio: float = 0.5) -> None:

        import matplotlib
        import matplotlib.pyplot as plt
        default_backend = plt.get_backend()

        try:
            import instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')

        # instaboost will modify the default backend
        # and cause visualization to fail.
        matplotlib.use(default_backend)

        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

    def _load_anns(self, results: dict) -> Tuple[list, list]:
        """Convert raw anns to instaboost expected input format."""
        anns = []
        ignore_anns = []
        for instance in results['instances']:
            label = instance['bbox_label']
            bbox = instance['bbox']
            mask = instance['mask']
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]

            if instance['ignore_flag'] == 0:
                anns.append({
                    'category_id': label,
                    'segmentation': mask,
                    'bbox': bbox
                })
            else:
                # Ignore instances without data augmentation
                ignore_anns.append(instance)
        return anns, ignore_anns

    def _parse_anns(self, results: dict, anns: list, ignore_anns: list,
                    img: np.ndarray) -> dict:
        """Restore the result of instaboost processing to the original anns
        format."""
        instances = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            instances.append(
                dict(
                    bbox=bbox,
                    bbox_label=ann['category_id'],
                    mask=ann['segmentation'],
                    ignore_flag=0))

        instances.extend(ignore_anns)
        results['img'] = img
        results['instances'] = instances
        return results

    def transform(self, results) -> dict:
        """The transform function."""
        img = results['img']
        ori_type = img.dtype
        if 'instances' not in results or len(results['instances']) == 0:
            return results

        anns, ignore_anns = self._load_anns(results)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            try:
                import instaboostfast as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            anns, img = instaboost.get_new_data(
                anns, img.astype(np.uint8), self.cfg, background=None)

        results = self._parse_anns(results, anns, ignore_anns,
                                   img.astype(ori_type))
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(aug_ratio={self.aug_ratio})'
        return repr_str
