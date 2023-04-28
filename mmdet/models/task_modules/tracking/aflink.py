# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import load_checkpoint
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from mmdet.registry import TASK_UTILS

INFINITY = 1e5


class TemporalBlock(BaseModule):
    """The temporal block of AFLink model.

    Args:
        in_channel (int): the dimension of the input channels.
        out_channel (int): the dimension of the output channels.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: tuple = (7, 1)):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bnf = nn.BatchNorm1d(out_channel)
        self.bnx = nn.BatchNorm1d(out_channel)
        self.bny = nn.BatchNorm1d(out_channel)

    def bn(self, x: Tensor) -> Tensor:
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0])
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1])
        x[:, :, :, 2] = self.bny(x[:, :, :, 2])
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FusionBlock(BaseModule):
    """The fusion block of AFLink model.

    Args:
        in_channel (int): the dimension of the input channels.
        out_channel (int): the dimension of the output channels.
    """

    def __init__(self, in_channel: int, out_channel: int):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(BaseModule):
    """The classifier of AFLink model.

    Args:
        in_channel (int): the dimension of the input channels.
    """

    def __init__(self, in_channel: int, out_channel: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_channel * 2, in_channel // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channel // 2, out_channel)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AFLinkModel(BaseModule):
    """Appearance-Free Link Model."""

    def __init__(self,
                 temporal_module_channels: list = [1, 32, 64, 128, 256],
                 fusion_module_channels: list = [256, 256],
                 classifier_channels: list = [256, 2]):
        super(AFLinkModel, self).__init__()
        self.TemporalModule_1 = nn.Sequential(*[
            TemporalBlock(temporal_module_channels[i],
                          temporal_module_channels[i + 1])
            for i in range(len(temporal_module_channels) - 1)
        ])

        self.TemporalModule_2 = nn.Sequential(*[
            TemporalBlock(temporal_module_channels[i],
                          temporal_module_channels[i + 1])
            for i in range(len(temporal_module_channels) - 1)
        ])

        self.FusionBlock_1 = FusionBlock(*fusion_module_channels)
        self.FusionBlock_2 = FusionBlock(*fusion_module_channels)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(*classifier_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        assert not self.training, 'Only testing is supported for AFLink.'
        x1 = x1[:, :, :, :3]
        x2 = x2[:, :, :, :3]
        x1 = self.TemporalModule_1(x1)  # [B,1,30,3] -> [B,256,6,3]
        x2 = self.TemporalModule_2(x2)
        x1 = self.FusionBlock_1(x1)
        x2 = self.FusionBlock_2(x2)
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1)
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1)
        y = self.classifier(x1, x2)
        y = torch.softmax(y, dim=1)[0, 1]
        return y


@TASK_UTILS.register_module()
class AppearanceFreeLink(BaseModule):
    """Appearance-Free Link method.

    This method is proposed in
    "StrongSORT: Make DeepSORT Great Again"
    `StrongSORT<https://arxiv.org/abs/2202.13514>`_.

    Args:
        checkpoint (str): Checkpoint path.
        temporal_threshold (tuple, optional): The temporal constraint
            for tracklets association. Defaults to (0, 30).
        spatial_threshold (int, optional): The spatial constraint for
            tracklets association. Defaults to 75.
        confidence_threshold (float, optional): The minimum confidence
            threshold for tracklets association. Defaults to 0.95.
    """

    def __init__(self,
                 checkpoint: str,
                 temporal_threshold: tuple = (0, 30),
                 spatial_threshold: int = 75,
                 confidence_threshold: float = 0.95):
        super(AppearanceFreeLink, self).__init__()
        self.temporal_threshold = temporal_threshold
        self.spatial_threshold = spatial_threshold
        self.confidence_threshold = confidence_threshold

        self.model = AFLinkModel()
        if checkpoint:
            load_checkpoint(self.model, checkpoint)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.fn_l2 = lambda x, y: np.sqrt(x**2 + y**2)

    def data_transform(self,
                       track1: np.ndarray,
                       track2: np.ndarray,
                       length: int = 30) -> Tuple[np.ndarray]:
        """Data Transformation. This is used to standardize the length of
        tracks to a unified length. Then perform min-max normalization to the
        motion embeddings.

        Args:
            track1 (ndarray): the first track with shape (N,C).
            track2 (ndarray): the second track with shape (M,C).
            length (int): the unified length of tracks. Defaults to 30.

        Returns:
            Tuple[ndarray]: the transformed track1 and track2.
        """
        # fill or cut track1
        length_1 = track1.shape[0]
        track1 = track1[-length:] if length_1 >= length else \
            np.pad(track1, ((length - length_1, 0), (0, 0)))

        # fill or cut track1
        length_2 = track2.shape[0]
        track2 = track2[:length] if length_2 >= length else \
            np.pad(track2, ((0, length - length_2), (0, 0)))

        # min-max normalization
        min_ = np.concatenate((track1, track2), axis=0).min(axis=0)
        max_ = np.concatenate((track1, track2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        track1 = (track1 - subtractor) / divisor
        track2 = (track2 - subtractor) / divisor

        return track1, track2

    def forward(self, pred_tracks: np.ndarray) -> np.ndarray:
        """Forward function.

        pred_tracks (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).

        Returns:
            ndarray: The linked tracks with shape (N, 7). Each row denotes
                (frame_id, track_id, x1, y1, x2, y2, score)
        """
        # sort tracks by the frame id
        pred_tracks = pred_tracks[np.argsort(pred_tracks[:, 0])]

        # gather tracks information
        id2info = defaultdict(list)
        for row in pred_tracks:
            frame_id, track_id, x1, y1, x2, y2 = row[:6]
            id2info[track_id].append([frame_id, x1, y1, x2 - x1, y2 - y1])
        id2info = {k: np.array(v) for k, v in id2info.items()}
        num_track = len(id2info)
        track_ids = np.array(list(id2info))
        cost_matrix = np.full((num_track, num_track), INFINITY)

        # compute the cost matrix
        for i, id_i in enumerate(track_ids):
            for j, id_j in enumerate(track_ids):
                if id_i == id_j:
                    continue
                info_i, info_j = id2info[id_i], id2info[id_j]
                frame_i, box_i = info_i[-1][0], info_i[-1][1:3]
                frame_j, box_j = info_j[0][0], info_j[0][1:3]
                # temporal constraint
                if not self.temporal_threshold[0] <= \
                        frame_j - frame_i <= self.temporal_threshold[1]:
                    continue
                # spatial constraint
                if self.fn_l2(box_i[0] - box_j[0], box_i[1] - box_j[1]) \
                        > self.spatial_threshold:
                    continue
                # confidence constraint
                track_i, track_j = self.data_transform(info_i, info_j)

                # numpy to torch
                track_i = torch.tensor(
                    track_i, dtype=torch.float).to(self.device)
                track_j = torch.tensor(
                    track_j, dtype=torch.float).to(self.device)
                track_i = track_i.unsqueeze(0).unsqueeze(0)
                track_j = track_j.unsqueeze(0).unsqueeze(0)

                confidence = self.model(track_i,
                                        track_j).detach().cpu().numpy()
                if confidence >= self.confidence_threshold:
                    cost_matrix[i, j] = 1 - confidence

        # linear assignment
        indices = linear_sum_assignment(cost_matrix)
        _id2id = dict()  # the temporary assignment results
        id2id = dict()  # the final assignment results
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < INFINITY:
                _id2id[i] = j
        for k, v in _id2id.items():
            if k in id2id:
                id2id[v] = id2id[k]
            else:
                id2id[v] = k

        # link
        for k, v in id2id.items():
            pred_tracks[pred_tracks[:, 1] == k, 1] = v

        # deduplicate
        _, index = np.unique(pred_tracks[:, :2], return_index=True, axis=0)

        return pred_tracks[index]
