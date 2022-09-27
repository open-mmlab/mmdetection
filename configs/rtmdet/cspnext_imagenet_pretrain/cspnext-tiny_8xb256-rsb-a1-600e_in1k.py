_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

backbone = dict(deepen_factor=0.33, widen_factor=0.5)
