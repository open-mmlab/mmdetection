from easydict import EasyDict as edict


_C = edict()
cfg = _C

# Set the config path relative to the main.py
_C.CONFIG = '../configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'
# Set the path to the candidate checkpoint relative to the main.py
_C.CHECKPOINT = '../checkpoints/mask_rcnn_r50_fpn_20e_compet_epoch_10.pth'

# post processing parameters
_C.RANDOM_ORDER = False  # when true, the value of c_order does not matter
_C.C_ORDER = True  # when true, read row first, otherwise, read column first
_C.THRESHOLDS = [.15, .35, .55]
_C.MIN_PIXELS = [75, 150, 75]

#allowed image formats
_C.ALLOWED_EXTENSIONS = {'png', 'tif'}

#show the resulting image with masks printed after prediction
_C.SHOW_RESULT = True