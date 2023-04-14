if '_base_':
    from .cspnext_s_8xb256_rsb_a1_600e_in1k import *

model.merge(
    dict(
        backbone=dict(deepen_factor=0.167, widen_factor=0.375),
        head=dict(in_channels=384)))
