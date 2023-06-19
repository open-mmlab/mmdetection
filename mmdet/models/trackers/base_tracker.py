# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from addict import Dict


class BaseTracker(metaclass=ABCMeta):
    """Base tracker model.

    Args:
        momentums (dict[str:float], optional): Momentums to update the buffers.
            The `str` indicates the name of the buffer while the `float`
            indicates the momentum. Defaults to None.
        num_frames_retain (int, optional). If a track is disappeared more than
            `num_frames_retain` frames, it will be deleted in the memo.
             Defaults to 10.
    """

    def __init__(self,
                 momentums: Optional[dict] = None,
                 num_frames_retain: int = 10) -> None:
        super().__init__()
        if momentums is not None:
            assert isinstance(momentums, dict), 'momentums must be a dict'
        self.momentums = momentums
        self.num_frames_retain = num_frames_retain

        self.reset()

    def reset(self) -> None:
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self) -> bool:
        """Whether the buffer is empty or not."""
        return False if self.tracks else True

    @property
    def ids(self) -> List[dict]:
        """All ids in the tracker."""
        return list(self.tracks.keys())

    @property
    def with_reid(self) -> bool:
        """bool: whether the framework has a reid model"""
        return hasattr(self, 'reid') and self.reid is not None

    def update(self, **kwargs) -> None:
        """Update the tracker.

        Args:
            kwargs (dict[str: Tensor | int]): The `str` indicates the
                name of the input variable. `ids` and `frame_ids` are
                obligatory in the keys.
        """
        memo_items = [k for k, v in kwargs.items() if v is not None]
        rm_items = [k for k in kwargs.keys() if k not in memo_items]
        for item in rm_items:
            kwargs.pop(item)
        if not hasattr(self, 'memo_items'):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items

        assert 'ids' in memo_items
        num_objs = len(kwargs['ids'])
        id_indice = memo_items.index('ids')
        assert 'frame_ids' in memo_items
        frame_id = int(kwargs['frame_ids'])
        if isinstance(kwargs['frame_ids'], int):
            kwargs['frame_ids'] = torch.tensor([kwargs['frame_ids']] *
                                               num_objs)
        # cur_frame_id = int(kwargs['frame_ids'][0])
        for k, v in kwargs.items():
            if len(v) != num_objs:
                raise ValueError('kwargs value must both equal')

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)

        self.pop_invalid_tracks(frame_id)

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['frame_ids'][-1] >= self.num_frames_retain:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def update_track(self, id: int, obj: Tuple[torch.Tensor]):
        """Update a track."""
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

    def init_track(self, id: int, obj: Tuple[torch.Tensor]):
        """Initialize a track."""
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                self.tracks[id][k] = v
            else:
                self.tracks[id][k] = [v]

    @property
    def memo(self) -> dict:
        """Return all buffers in the tracker."""
        outs = Dict()
        for k in self.memo_items:
            outs[k] = []

        for id, objs in self.tracks.items():
            for k, v in objs.items():
                if k not in outs:
                    continue
                if self.momentums is not None and k in self.momentums:
                    v = v
                else:
                    v = v[-1]
                outs[k].append(v)

        for k, v in outs.items():
            outs[k] = torch.cat(v, dim=0)
        return outs

    def get(self,
            item: str,
            ids: Optional[list] = None,
            num_samples: Optional[int] = None,
            behavior: Optional[str] = None) -> torch.Tensor:
        """Get the buffer of a specific item.

        Args:
            item (str): The demanded item.
            ids (list[int], optional): The demanded ids. Defaults to None.
            num_samples (int, optional): Number of samples to calculate the
                results. Defaults to None.
            behavior (str, optional): Behavior to calculate the results.
                Options are `mean` | None. Defaults to None.

        Returns:
            Tensor: The results of the demanded item.
        """
        if ids is None:
            ids = self.ids

        outs = []
        for id in ids:
            out = self.tracks[id][item]
            if isinstance(out, list):
                if num_samples is not None:
                    out = out[-num_samples:]
                    out = torch.cat(out, dim=0)
                    if behavior == 'mean':
                        out = out.mean(dim=0, keepdim=True)
                    elif behavior is None:
                        out = out[None]
                    else:
                        raise NotImplementedError()
                else:
                    out = out[-1]
            outs.append(out)
        return torch.cat(outs, dim=0)

    @abstractmethod
    def track(self, *args, **kwargs):
        """Tracking forward function."""
        pass

    def crop_imgs(self,
                  img: torch.Tensor,
                  meta_info: dict,
                  bboxes: torch.Tensor,
                  rescale: bool = False) -> torch.Tensor:
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.

        Args:
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
            meta_info (dict): image information dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the scale of the image. Defaults to False.

        Returns:
            Tensor: Image tensor of shape (T, C, H, W).
        """
        h, w = meta_info['img_shape']
        img = img[:, :, :h, :w]
        if rescale:
            factor_x, factor_y = meta_info['scale_factor']
            bboxes[:, :4] *= torch.tensor(
                [factor_x, factor_y, factor_x, factor_y]).to(bboxes.device)
        bboxes[:, 0] = torch.clamp(bboxes[:, 0], min=0, max=w - 1)
        bboxes[:, 1] = torch.clamp(bboxes[:, 1], min=0, max=h - 1)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], min=1, max=w)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], min=1, max=h)

        crop_imgs = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            crop_img = img[:, :, y1:y2, x1:x2]
            if self.reid.get('img_scale', False):
                crop_img = F.interpolate(
                    crop_img,
                    size=self.reid['img_scale'],
                    mode='bilinear',
                    align_corners=False)
            crop_imgs.append(crop_img)

        if len(crop_imgs) > 0:
            return torch.cat(crop_imgs, dim=0)
        elif self.reid.get('img_scale', False):
            _h, _w = self.reid['img_scale']
            return img.new_zeros((0, 3, _h, _w))
        else:
            return img.new_zeros((0, 3, h, w))
