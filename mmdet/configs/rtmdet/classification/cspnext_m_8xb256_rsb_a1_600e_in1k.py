if '_base_':
    from .cspnext_s_8xb256_rsb_a1_600e_in1k import *

model.merge(
    dict(
        backbone=dict(deepen_factor=0.67, widen_factor=0.75),
        head=dict(in_channels=768)))
