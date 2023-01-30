_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    head=dict(in_channels=768))
