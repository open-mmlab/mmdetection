# Copyright (c) OpenMMLab. All rights reserved.
# Related to Theorem 2.


def get_detection_weight(n):
    a = (n.sum() - n) / n
    w = a * ((1 + a) / a).log()

    return w[None]
