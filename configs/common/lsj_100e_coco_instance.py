_base_ = './lsj_50e_coco_instance.py'

# simply change the repeat time from 2 to 4 for 100 e training
data = dict(train=dict(times=4))
