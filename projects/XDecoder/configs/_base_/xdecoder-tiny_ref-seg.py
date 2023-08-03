_base_ = 'xdecoder-tiny_open-vocab-semseg.py'

model = dict(head=dict(task='ref-seg'))
