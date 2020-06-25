from mmcv.cnn import NORM_LAYERS
from mmcv.ops import SyncBatchNorm

NORM_LAYERS.register_module(module=SyncBatchNorm, name='MMSyncBN', force=True)
