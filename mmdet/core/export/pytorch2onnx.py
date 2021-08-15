from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint


def generate_inputs_and_wrap_model(config_path,
                                   checkpoint_path,
                                   input_config,
                                   cfg_options=None):
    """Prepare sample input and wrap model for ONNX export.

    The ONNX export API only accept args, and all inputs should be
    torch.Tensor or corresponding types (such as tuple of tensor).
    So we should call this function before exporting. This function will:

    1. generate corresponding inputs which are used to execute the model.
    2. Wrap the model's forward function.

    For example, the MMDet models' forward function has a parameter
    ``return_loss:bool``. As we want to set it as False while export API
    supports neither bool type or kwargs. So we have to replace the forward
    method like ``model.forward = partial(model.forward, return_loss=False)``.

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint
        input_config (dict): the exactly data in this dict depends on the
            framework. For MMSeg, we can just declare the input shape,
            and generate the dummy data accordingly. However, for MMDet,
            we may pass the real img path, or the NMS will return None
            as there is no legal bbox.

    Returns:
        tuple: (model, tensor_data) wrapped model which can be called by
            ``model(*tensor_data)`` and a list of inputs which are used to
            execute the model while exporting.
    """

    model = build_model_from_cfg(
        config_path, checkpoint_path, cfg_options=cfg_options)
    one_img, one_meta = preprocess_example_input(input_config)
    tensor_data = [one_img]
    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss=False)

    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    opset_version = 11
    # put the import within the function thus it will not cause import error
    # when not using this function
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(opset_version)

    return model, tensor_data


def build_model_from_cfg(config_path, checkpoint_path, cfg_options=None):
    """Build a model from config and load the given checkpoint.

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint

    Returns:
        torch.nn.Module: the built model
    """
    from mmdet.models import build_detector

    cfg = mmcv.Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmdet.datasets import DATASETS
        dataset = DATASETS.get(cfg.data.test['type'])
        assert (dataset is not None)
        model.CLASSES = dataset.CLASSES
    model.cpu().eval()
    return model


def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and \
            meta information for the example input image.

    Examples:
        >>> from mmdet.core.export import preprocess_example_input
        >>> input_config = {
        >>>         'input_shape': (1,3,224,224),
        >>>         'input_path': 'demo/demo.jpg',
        >>>         'normalize_cfg': {
        >>>             'mean': (123.675, 116.28, 103.53),
        >>>             'std': (58.395, 57.12, 57.375)
        >>>             }
        >>>         }
        >>> one_img, one_meta = preprocess_example_input(input_config)
        >>> print(one_img.shape)
        torch.Size([1, 3, 224, 224])
        >>> print(one_meta)
        {'img_shape': (224, 224, 3),
        'ori_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False}
    """
    input_path = input_config['input_path']
    input_shape = input_config['input_shape']
    one_img = mmcv.imread(input_path)
    one_img = mmcv.imresize(one_img, input_shape[2:][::-1])
    show_img = one_img.copy()
    if 'normalize_cfg' in input_config.keys():
        normalize_cfg = input_config['normalize_cfg']
        mean = np.array(normalize_cfg['mean'], dtype=np.float32)
        std = np.array(normalize_cfg['std'], dtype=np.float32)
        to_rgb = normalize_cfg.get('to_rgb', True)
        one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=to_rgb)
    one_img = one_img.transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(
        True)
    (_, C, H, W) = input_shape
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.ones(4, dtype=np.float32),
        'flip': False,
        'show_img': show_img,
        'flip_direction': None
    }

    return one_img, one_meta
