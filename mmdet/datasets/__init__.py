# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .base_video_dataset import BaseVideoDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .dsdl import DSDLDetDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .mot_challenge_dataset import MOTChallengeDataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .reid_dataset import ReIDDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler,
                       TrackAspectRatioBatchSampler, TrackImgSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .youtube_vis_dataset import YouTubeVISDataset

__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset', 'DSDLDetDataset',
    'BaseVideoDataset', 'MOTChallengeDataset', 'TrackImgSampler',
    'ReIDDataset', 'YouTubeVISDataset', 'TrackAspectRatioBatchSampler'
]
