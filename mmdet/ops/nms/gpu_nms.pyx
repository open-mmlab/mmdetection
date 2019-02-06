# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int, size_t) nogil
    size_t nms_Malloc() nogil

memory_pool = {}

def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = 5
    cdef int num_out
    cdef size_t base
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 4]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :5]
    cdef float cthresh = thresh
    if device_id not in memory_pool:
        with nogil:
            base = nms_Malloc()
        memory_pool[device_id] = base
        # print "malloc", base
    base = memory_pool[device_id]
    with nogil:
        _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, cthresh, device_id, base)
    keep = keep[:num_out]
    return list(order[keep])
