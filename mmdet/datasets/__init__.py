# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .builder import DATASETS, PIPELINES, build_dataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset
from .dataset_wrappers import MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'DATASETS', 'PIPELINES', 'build_dataset',
    'get_loading_pipeline', 'CocoPanopticDataset', 'MultiImageMixDataset',
    'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset'
]
