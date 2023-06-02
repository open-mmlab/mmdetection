# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS
from ..evaluation import get_classes
from ..registry import MODELS
from ..structures import DetDataSample, SampleList
from ..utils import get_test_pipeline_cfg


<<<<<<< HEAD
def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """从配置文件初始化检测模型.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): 配置文件路径,或配置对象.
        checkpoint (str, optional): 权重路径. 如果为None, 模型不会加载任何权重.
        cfg_options (dict): 覆盖所用配置中的某些设置的选项.
=======
def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
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
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
>>>>>>> mmdetection/main

    Returns:
        nn.Module: 初始化后的检测模型.
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
<<<<<<< HEAD
    config.model.train_cfg = None  # 避免构建只在train才使用的方法,assigner, sampler等
    model = build_detector(config.model)
    if checkpoint is not None:
=======
    init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
>>>>>>> mmdetection/main
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
<<<<<<< HEAD
            warnings.warn('加载的权重中不存在类别列表,默认使用COCO类.')
            model.CLASSES = get_classes('coco')  # 初始化为coco的80类
    model.cfg = config  # 为方便起见,将配置保存在模型中
=======
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
>>>>>>> mmdetection/main
    model.to(device)
    model.eval()

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config

    return model


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


<<<<<<< HEAD
def inference_detector(model, imgs):
    """使用检测模型推理单(多)幅图像.

    Args:
        model (nn.Module): 已加载的检测器模型.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           图像文件路径或已加载的numpy数据,可为单个str/np 或者[str/np,]/(str/np,).

    Returns:
        如果 imgs 是列表或元组, 则返回相同长度的列表类型结果, 否则直接返回检测结果.
=======
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
>>>>>>> mmdetection/main
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
<<<<<<< HEAD
        # 如果参数img是numpy型数据,那么切换数据加载方式
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
=======
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
>>>>>>> mmdetection/main

        test_pipeline = Compose(test_pipeline)

<<<<<<< HEAD
    datas = []
    for img in imgs:
        # 准备数据,当img为numpy数据时,代表它已是无需加载的
        if isinstance(img, np.ndarray):
            # 直接添加img
            data = dict(img=img)
        else:
            # 否则需要通过LoadImageFromFile来从路径中加载图片
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
    # 将多张图片整合到一起, 包括兼容不同长宽比的图片至同一尺寸
    # 注! 它是将所有测试图片揉到一个batch中,只跑一遍inference.
    data = collate(datas, samples_per_gpu=len(imgs))
    # {'img_metas':[DC([]),], 'img':[DC[tensor(bs, 3, 480, 640),],]}
    # 最外面的[]是TTA数量,里面的[]中的数据是GPU数量的数据,TTA与bs不能同时大于1
    # inference_detector 默认单卡推理,所以DC包裹的列表长度为1
    # 此处的.data[0]意为从DataContainer(.data)获取第一张卡([0])的数据
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # 将数据分配到指定的 GPU
        data = scatter(data, [device])[0]
    else:
=======
    if model.data_preprocessor.device.type == 'cpu':
>>>>>>> mmdetection/main
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

<<<<<<< HEAD
    # 对模型前向传播
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
=======
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
>>>>>>> mmdetection/main

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
<<<<<<< HEAD


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """在图像上绘制检测结果.

    Args:
        model (nn.Module): 加载的检测模型.
        img (str or np.ndarray): 图像文件名或加载的图像数据.
        result (tuple[list] or list): 检测结果,(bbox, segm) or bbox
        score_thr (float): boxes 和 mask 的置信度显示阈值.
        title (str): 图像展示窗口的标题.
        wait_time (float): waitKey 参数的值. 默认为: 0ms.意为无限等待
        palette (str or tuple(int) or :obj:`Color`): 颜色, 如果是元组应按 BGR 顺序.
        out_file (str or None): 图片保存的路径.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)
=======
>>>>>>> mmdetection/main
