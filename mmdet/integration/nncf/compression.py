import pathlib
import torch
from functools import partial

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
    except ImportError:
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
    try:
        checkpoint = torch.load(path)
        meta = checkpoint.get('meta', {})
        nncf_enable_compression = meta.get('nncf_enable_compression', False)
        return bool(nncf_enable_compression)
    except FileNotFoundError:
        return False


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
    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)
    logger = get_root_logger(cfg.log_level)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        nncf_config = register_default_init_args(nncf_config, None, wrapped_loader)

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
        return get_fake_input_func(cfg, orig_img_shape=tuple([H, W, C]), device=device)

    def dummy_forward(model):
        fake_data = _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        img = nncf_model_input(img)
        if nncf_compress_postprocessing:
            ctx = model.forward_export_context(img_metas)
            logger.debug(f"NNCF will compress a postprocessing part of the model")
        else:
            ctx = model.forward_dummy_context(img_metas)
            logger.debug(f"NNCF will NOT compress a postprocessing part of the model")
        with ctx:
            model(img)

    model.dummy_forward_fn = dummy_forward

    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      resuming_state_dict=resuming_state_dict)
    model = change_export_func_first_conv(model)

    return compression_ctrl, model


def change_export_func_first_conv(model):
    ''' To avoid saturation issue
    At the moment works only for mobilenet
    '''

    def run_hacked_export_quantization(self, x):
        from nncf.quantization.layers import (
            ExportQuantizeToFakeQuantize, ExportQuantizeToONNXQuantDequant,
            QuantizerExportMode, get_scale_zp_from_input_low_input_high)
        from nncf.utils import no_jit_trace
        with no_jit_trace():
            input_range = abs(self.scale) + self.eps
            # todo: take bias into account during input_low/input_high calculation
            input_low = input_range * self.level_low / self.level_high
            input_high = input_range

            if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
                y_scale, y_zero_point = get_scale_zp_from_input_low_input_high(self.level_low,
                                                                               self.level_high,
                                                                               input_low,
                                                                               input_high)

        if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
            return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
        if self._export_mode == QuantizerExportMode.FAKE_QUANTIZE:
            x = x / 2.0
            return ExportQuantizeToFakeQuantize.apply(x, self.levels, input_low, input_high, input_low * 2,
                                                      input_high * 2)
        raise RuntimeError

    logger = get_root_logger()
    orig_model = model.get_nncf_wrapped_model()
    try:
        # pylint: disable=protected-access
        module_ = orig_model.backbone.features.init_block.conv.pre_ops._modules['0']
    except AttributeError as e:
        logger.info(f'Cannot change an export function for the first Conv due  {e}')
        return model
    module_.op.run_export_quantization = partial(run_hacked_export_quantization, module_.op)
    logger.info('Change an export function for the first Conv to avoid saturation issue on AVX2, AVX512')
    return model


def get_uncompressed_model(module):
    if not is_nncf_enabled():
        return module
    if isinstance(module, NNCFNetwork):
        return module.get_nncf_wrapped_model()
    return module
