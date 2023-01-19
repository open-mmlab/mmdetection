# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from .text_encoder import CLIPTextEncoder


def get_text_embeddings(dataset=None,
                        custom_vocabulary=None,
                        prompt_prefix='a '):
    assert (dataset is None) ^ (custom_vocabulary is None), \
        'Either `dataset` or `custom_vocabulary` should be specified.'
    if dataset:
        assert dataset in DATASET_EMBEDDINGS
        return DATASET_EMBEDDINGS[dataset]
    elif custom_vocabulary:
        text_encoder = CLIPTextEncoder()
        text_encoder.eval()
        texts = [prompt_prefix + x for x in custom_vocabulary]
        embeddings = text_encoder(texts).detach().permute(
            1, 0).contiguous().cpu()
        return embeddings
    else:
        raise NotImplementedError


DATASET_EMBEDDINGS = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}


def get_class_names(dataset):
    if dataset == 'coco':
        from mmdet.datasets import CocoDataset
        class_names = CocoDataset.METAINFO['classes']
    elif dataset == 'cityscapes':
        from mmdet.datasets import CityscapesDataset
        class_names = CityscapesDataset.METAINFO['classes']
    elif dataset == 'voc':
        from mmdet.datasets import VOCDataset
        class_names = VOCDataset.METAINFO['classes']
    elif dataset == 'openimages':
        from mmdet.datasets import OpenImagesDataset
        class_names = OpenImagesDataset.METAINFO['classes']
    elif dataset == 'lvis':
        from mmdet.datasets import LVISV1Dataset
        class_names = LVISV1Dataset.METAINFO['classes']
    else:
        raise TypeError(f'Invalid type for dataset name: {type(dataset)}')
    return class_names


def reset_cls_layer_weight(model, weight):
    if type(weight) == str:
        print('Resetting cls_layer_weight from file: ', weight)
        zs_weight = torch.tensor(
            np.load(weight),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
    else:
        zs_weight = weight
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros(
            (zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to('cuda')
    num_classes = zs_weight.shape[-1]

    for bbox_head in model.roi_head.bbox_head:
        bbox_head.num_classes = num_classes
        del bbox_head.fc_cls.zs_weight
        bbox_head.fc_cls.zs_weight = zs_weight
