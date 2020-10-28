import pathlib
import torch

from mmdet.utils import get_root_logger
from .utils import (check_nncf_is_enabled, get_nncf_version, is_nncf_enabled,
                    load_checkpoint)

if is_nncf_enabled():
    try:
        from nncf import (NNCFConfig, create_compressed_model,
                          register_default_init_args)
        from nncf.dynamic_graph.patch_pytorch import nncf_model_input
        from nncf.initialization import InitializingDataLoader
        from nncf.nncf_network import NNCFNetwork

        class_InitializingDataLoader = InitializingDataLoader
    except:  # noqa: E722
        raise RuntimeError(
            'Cannot import the standard functions of NNCF library '
            '-- most probably, incompatible version of NNCF. '
            'Please, use NNCF version pointed in the documentation.')
else:

    class DummyInitializingDataLoader:
        pass

    class_InitializingDataLoader = DummyInitializingDataLoader


class MMInitializeDataLoader(class_InitializingDataLoader):
    def get_inputs(self, dataloader_output):
        # redefined InitializingDataLoader because
        # of DataContainer format in mmdet
        kwargs = {k: v.data[0] for k, v in dataloader_output.items()}
        return (), kwargs

    # TODO: not tested; need to test
    def get_target(self, dataloader_output):
        return dataloader_output['gt_bboxes'], dataloader_output['gt_labels']


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
    checkpoint = torch.load(path)
    meta = checkpoint.get('meta', {})
    nncf_enable_compression = meta.get('nncf_enable_compression', False)
    return bool(nncf_enable_compression)


def wrap_nncf_model(model,
                    cfg,
                    data_loader_for_init=None,
                    get_fake_input_func=None,
                    should_compress_postprocessing=True):
    """
    The function wraps mmdet model by NNCF
    Note that the parameter `get_fake_input_func` should be the function `get_fake_input`
    -- cannot import this function here explicitly
    """
    check_nncf_is_enabled()
    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)
    logger = get_root_logger(cfg.log_level)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        nncf_config = register_default_init_args(nncf_config, wrapped_loader)

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

    def _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        assert get_fake_input_func is not None

        input_size = nncf_config.get('input_info').get('sample_size')
        assert len(input_size) == 4 and input_size[0] == 1

        H, W = input_size[-2:]
        C = input_size[1]
        orig_img_shape = tuple([H, W, C])  # HWC order here for np.zeros to emulate cv2.imread

        device = next(model.parameters()).device

        # NB: the full cfg is required here!
        fake_data = get_fake_input_func(cfg,
                                        orig_img_shape=orig_img_shape,
                                        device=device)
        return fake_data

    def dummy_forward_without_export_part(model):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        fake_data = _get_fake_data_for_forward(cfg,
                                               nncf_config,
                                               get_fake_input_func)
        img, img_metas = fake_data['img'], fake_data['img_metas']
        img = nncf_model_input(img)
        with model.forward_dummy_context(img_metas):
            model(img)

    def dummy_forward_with_export_part(model):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        fake_data = _get_fake_data_for_forward(cfg, nncf_config,
                                               get_fake_input_func)
        img, img_metas = fake_data['img'], fake_data['img_metas']
        img = nncf_model_input(img)
        with model.forward_export_context(img_metas):
            model(img)

    if 'nncf_should_compress_postprocessing' in cfg:
        # NB: This parameter is used to choose if we should try to make NNCF compression
        #     for a whole model graph including postprocessing (`dummy_forward_with_export_part`),
        #     or make NNCF compression of the part of the model without postprocessing
        #     (`dummy_forward_without_export_part`).
        #     Our primary goal is to make NNCF compression of such big part of the model as
        #     possible, so `dummy_forward_with_export_part` is our primary choice, whereas
        #     `dummy_forward_without_export_part` is our fallback decision.
        #     When we manage to enable NNCF compression for sufficiently many models,
        #     we should keep one choice only.
        should_compress_postprocessing = \
                cfg.get('nncf_should_compress_postprocessing')
        logger.debug('set should_compress_postprocessing='
                     f'{should_compress_postprocessing}')

    if should_compress_postprocessing:
        logger.debug('dummy_forward = dummy_forward_with_export_part')
        dummy_forward = dummy_forward_with_export_part
    else:
        logger.debug('dummy_forward = dummy_forward_without_export_part')
        dummy_forward = dummy_forward_without_export_part

    model.dummy_forward_fn = dummy_forward

    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      resuming_state_dict=resuming_state_dict)
    return compression_ctrl, model


def get_uncompressed_model(module):
    if not is_nncf_enabled():
        return module
    if isinstance(module, NNCFNetwork):
        return module.get_nncf_wrapped_model()
    return module
