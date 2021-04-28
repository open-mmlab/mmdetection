import os
import pathlib
import tempfile

import mmcv
import torch

from mmdet.utils import get_root_logger
from .utils import (check_nncf_is_enabled, get_nncf_version, is_nncf_enabled,
                    load_checkpoint, no_nncf_trace)


def get_nncf_metadata():
    """
    The function returns NNCF metadata that should be stored into a checkpoint.
    The metadata is used to check in wrap_nncf_model if the checkpoint should be used
    to resume NNCF training or initialize NNCF fields of NNCF-wrapped model.
    """
    check_nncf_is_enabled()
    return dict(nncf_enable_compression=True, nncf_version=get_nncf_version())


def is_checkpoint_nncf(path):
    """
    The function uses metadata stored in a checkpoint to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    See the function get_nncf_metadata above.
    """
    try:
        checkpoint = torch.load(path, map_location='cpu')
        meta = checkpoint.get('meta', {})
        nncf_enable_compression = meta.get('nncf_enable_compression', False)
        return bool(nncf_enable_compression)
    except FileNotFoundError:
        return False


def get_nncf_config_from_meta(path):
    """
    The function uses metadata stored in a checkpoint to restore the nncf
    part of the model config.
    """
    logger = get_root_logger()
    checkpoint = torch.load(path, map_location='cpu')
    meta = checkpoint.get('meta', {})

    nncf_enable_compression = meta.get('nncf_enable_compression', False)
    assert nncf_enable_compression, \
        'get_nncf_config_from_meta should be run for NNCF-compressed checkpoints only'

    config_text = meta['config']

    with tempfile.NamedTemporaryFile(prefix='config_', suffix='.py',
                                     mode='w', delete=False) as f_tmp:
        f_tmp.write(config_text)
        tmp_name = f_tmp.name
    cfg = mmcv.Config.fromfile(tmp_name)
    os.unlink(tmp_name)

    nncf_config = cfg.get('nncf_config')

    assert isinstance(nncf_config, dict), (
        f'Wrong nncf_config part of the config saved in the metainfo'
        f' of the snapshot {path}:'
        f' nncf_config={nncf_config}')

    nncf_config_part = {
        'nncf_config': nncf_config,
        'find_unused_parameters': True
    }
    if nncf_config_part['nncf_config'].get('log_dir'):
        # TODO(LeonidBeynenson): improve work with log dir
        log_dir = tempfile.mkdtemp(prefix='nncf_output_')
        nncf_config_part['nncf_config']['log_dir'] = log_dir

    logger.info(f'Read nncf config from meta nncf_config_part={nncf_config_part}')
    return nncf_config_part


def wrap_nncf_model(model,
                    cfg,
                    data_loader_for_init=None,
                    get_fake_input_func=None):
    """
    The function wraps mmdet model by NNCF
    Note that the parameter `get_fake_input_func` should be the function `get_fake_input`
    -- cannot import this function here explicitly
    """

    check_nncf_is_enabled()

    from nncf import (NNCFConfig, create_compressed_model,
                      register_default_init_args)
    from nncf.dynamic_graph.io_handling import nncf_model_input
    from nncf.dynamic_graph.trace_tensor import TracedTensor
    from nncf.initialization import InitializingDataLoader

    class MMInitializeDataLoader(InitializingDataLoader):
        def get_inputs(self, dataloader_output):
            # redefined InitializingDataLoader because
            # of DataContainer format in mmdet
            kwargs = {k: v.data[0] for k, v in dataloader_output.items()}
            return (), kwargs

    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)
    logger = get_root_logger(cfg.log_level)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        nncf_config = register_default_init_args(nncf_config, wrapped_loader, device=next(model.parameters()).device)

    if cfg.get('resume_from'):
        checkpoint_path = cfg.get('resume_from')
        assert is_checkpoint_nncf(checkpoint_path), (
            'It is possible to resume training with NNCF compression from NNCF checkpoints only. '
            'Use "load_from" with non-compressed model for further compression by NNCF.')
    elif cfg.get('load_from'):
        checkpoint_path = cfg.get('load_from')
        if not is_checkpoint_nncf(checkpoint_path):
            checkpoint_path = None
            logger.info('Received non-NNCF checkpoint to start training '
                        '-- initialization of NNCF fields will be done')
    else:
        checkpoint_path = None

    if not data_loader_for_init and not checkpoint_path:
        raise RuntimeError('Either data_loader_for_init or NNCF pre-trained '
                           'model checkpoint should be set')

    if checkpoint_path:
        logger.info(f'Loading NNCF checkpoint from {checkpoint_path}')
        logger.info(
            'Please, note that this first loading is made before addition of '
            'NNCF FakeQuantize nodes to the model, so there may be some '
            'warnings on unexpected keys')
        resuming_state_dict = load_checkpoint(model, checkpoint_path)
        logger.info(f'Loaded NNCF checkpoint from {checkpoint_path}')
    else:
        resuming_state_dict = None

    if "nncf_compress_postprocessing" in cfg:
        # NB: This parameter is used to choose if we should try to make NNCF compression
        #     for a whole model graph including postprocessing (`nncf_compress_postprocessing=True`),
        #     or make NNCF compression of the part of the model without postprocessing
        #     (`nncf_compress_postprocessing=False`).
        #     Our primary goal is to make NNCF compression of such big part of the model as
        #     possible, so `nncf_compress_postprocessing=True` is our primary choice, whereas
        #     `nncf_compress_postprocessing=False` is our fallback decision.
        #     When we manage to enable NNCF compression for sufficiently many models,
        #     we should keep one choice only.
        nncf_compress_postprocessing = cfg.get('nncf_compress_postprocessing')
        logger.debug('set should_compress_postprocessing='f'{nncf_compress_postprocessing}')
    else:
        nncf_compress_postprocessing = True

    def _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func):
        input_size = nncf_config.get("input_info").get('sample_size')
        assert get_fake_input_func is not None
        assert len(input_size) == 4 and input_size[0] == 1
        H, W, C = input_size[2], input_size[3], input_size[1]
        device = next(model.parameters()).device
        with no_nncf_trace():
            return get_fake_input_func(cfg, orig_img_shape=tuple([H, W, C]), device=device)

    def dummy_forward(model):
        fake_data = _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        img[0] = nncf_model_input(img[0])
        if nncf_compress_postprocessing:
            ctx = model.forward_export_context(img_metas)
            logger.debug(f"NNCF will compress a postprocessing part of the model")
        else:
            ctx = model.forward_dummy_context(img_metas)
            logger.debug(f"NNCF will NOT compress a postprocessing part of the model")
        with ctx:
            model(img)

    def wrap_inputs(args, kwargs):
        # during dummy_forward
        if not len(kwargs):
            if not isinstance(args[0][0], TracedTensor):
                args[0][0] = nncf_model_input(args[0][0])
            return args, kwargs

        # during building original graph
        if not kwargs.get('return_loss') and kwargs.get('forward_export'):
            return args, kwargs

        # during model's forward in export
        assert 'img' in kwargs, 'During model forward img must be in kwargs'
        img = kwargs['img']
        if isinstance(img, list):
            assert len(img) == 1, 'Input list must have a length 1'
            assert torch.is_tensor(img[0]), 'Input for a model must be a tensor'
            img[0] = nncf_model_input(img[0])
        else:
            assert torch.is_tensor(img), 'Input for a model must be a tensor'
            img = nncf_model_input(img)
        kwargs['img'] = img
        return args, kwargs

    model.dummy_forward_fn = dummy_forward
    export_method = type(model).export

    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      resuming_state_dict=resuming_state_dict)
    model.export = export_method.__get__(model)

    return compression_ctrl, model


def get_uncompressed_model(module):
    if not is_nncf_enabled():
        return module
    from nncf.nncf_network import NNCFNetwork
    if isinstance(module, NNCFNetwork):
        return module.get_nncf_wrapped_model()
    return module
