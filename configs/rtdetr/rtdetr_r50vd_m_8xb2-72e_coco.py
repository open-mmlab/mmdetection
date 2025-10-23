_base_ = './rtdetr_r50vd_8xb2-72e_coco.py'

model = dict(eval_idx=1)  # use 2th decoder layer to eval
