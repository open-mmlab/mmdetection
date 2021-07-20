#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

_base_ = './yolox_s_inference_coco.py'

model = dict(
    backbone=dict(dep_mul=1.33, wid_mul=1.25),
    neck=dict(depth=1.33, width=1.25),
    bbox_head=dict(width=1.25))
