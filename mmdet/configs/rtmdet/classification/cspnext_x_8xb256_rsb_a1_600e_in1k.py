if '_base_':
    from .cspnext_s_8xb256_rsb_a1_600e_in1k import *

model.merge(
    dict(
        backbone=dict(deepen_factor=1.33, widen_factor=1.25),
        head=dict(in_channels=1280)))
