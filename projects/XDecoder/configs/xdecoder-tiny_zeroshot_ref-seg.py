_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg_ade20k.py'

model = dict(head=dict(task='ref-seg'))
