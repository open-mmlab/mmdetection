# import mmcv
# import numpy as np
# import torch
#
# from mmdet.models import YOLOV4, build_detector
#
#
# def _build_model_from_cfg(config_path):
#     """Build a model from config and load the given checkpoint.
#
#     Args:
#         config_path (str): the OpenMMLab config for the model we want to
#         export to ONNX
#         checkpoint_path (str): Path to the corresponding checkpoint
#     Returns:
#         torch.nn.Module: the built model
#     """
#
#     cfg = mmcv.Config.fromfile(config_path)
#     # import modules from string list.
#     if cfg.get('custom_imports', None):
#         from mmcv.utils import import_modules_from_strings
#         import_modules_from_strings(**cfg['custom_imports'])
#     cfg.model.pretrained = None
#     cfg.data.test.test_mode = True
#
#     # build the model
#     model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
#     # load_checkpoint(model, checkpoint_path, map_location='cpu')
#     # model.cpu().eval()
#     return model
#
#
# def _preprocess_example_input(input_config):
#     """Prepare an example input image for `generate_inputs_and_wrap_model`.
#     Args:
#         input_config (dict): customized config describing the example input.
#         Example:
#         input_config: {
#             'input_shape':[1,3,224,224],
#             'input_path': 'demo/demo.jpg',
#             'normalize_cfg': {
#                 'mean': [123.675, 116.28, 103.53],
#                 'std': [58.395, 57.12, 57.375]
#             }
#         }
#     Returns:
#         tuple: (one_img, one_meta), tensor of the example input image and
#         meta
#         information for the example input image.
#     """
#     input_path = input_config['input_path']
#     input_shape = input_config['input_shape']
#     one_img = mmcv.imread(input_path)
#     if 'normalize_cfg' in input_config.keys():
#         normalize_cfg = input_config['normalize_cfg']
#         mean = np.array(normalize_cfg['mean'], dtype=np.float32)
#         std = np.array(normalize_cfg['std'], dtype=np.float32)
#         one_img = mmcv.imnormalize(one_img, mean, std)
#     one_img = mmcv.imresize(one_img, input_shape[2:]).transpose(2, 0, 1)
#     one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(
#         True)
#     (_, C, H, W) = input_shape
#     one_meta = {
#         'img_shape': (H, W, C),
#         'ori_shape': (H, W, C),
#         'pad_shape': (H, W, C),
#         'filename': '<demo>.png',
#         'scale_factor': 1.0,
#         'flip': False
#     }
#
#     return one_img, one_meta
#
#
# input_config = {
#     'input_shape': [1, 3, 224, 224],
#     'input_path': 'demo/demo.jpg',
#     'normalize_cfg': {
#         'mean': [123.675, 116.28, 103.53],
#         'std': [58.395, 57.12, 57.375]
#     }
# }
# model = _build_model_from_cfg('configs/yolo/yolov4_spp_new_param.py')
# model.train()
# one_img, one_meta = _preprocess_example_input(input_config)
# result = model([one_img], [[one_meta]], return_loss=True)
