# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from ..evaluation import get_classes
from ..models import build_detector
from ..structures import DetDataSample, SampleList


def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'coco',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to coco.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x
            model.dataset_meta = checkpoint_meta['dataset_meta']
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'CLASSES': classes, 'PALETTE': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {
                'CLASSES': get_classes('coco'),
                'PALETTE': palette
            }

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = cfg.test_dataloader.dataset.pipeline
        if isinstance(imgs[0], np.ndarray):
            # set loading pipeline type
            test_pipeline[0].type = 'LoadImageFromNDArray'

        new_test_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] != 'LoadAnnotations' and pipeline[
                    'type'] != 'LoadPanopticAnnotations':
                new_test_pipeline.append(pipeline)

        test_pipeline = Compose(new_test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


# TODO: Awaiting refactoring
async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNDArray'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    for m in model.modules():
        assert not isinstance(
            m,
            RoIPool), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(data, rescale=True)
    return results
