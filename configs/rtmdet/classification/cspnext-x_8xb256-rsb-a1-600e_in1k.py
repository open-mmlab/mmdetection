_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    head=dict(in_channels=1280))
