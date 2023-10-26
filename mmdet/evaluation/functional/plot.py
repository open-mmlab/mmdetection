# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import numpy as np
from mmengine.visualization import Visualizer


def prplot(rs, ps, class_name, iou_type, types):

    def fig2im(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype='u1')
        im = buf_ndarray.reshape(h, w, 3)
        return im

    vis = Visualizer.get_current_instance()
    cs = np.vstack([
        np.array([0.31, 0.51, 0.74]),
        np.array([0.75, 0.31, 0.30]),
        np.array([0.36, 0.90, 0.38]),
        np.array([0.50, 0.39, 0.64]),
        np.array([1, 0.6, 0]),
    ])

    figure_title = iou_type + '-' + class_name
    ps_curve = [ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in ps]
    fig = plt.figure()
    ax = plt.subplot(111)
    for k in range(len(types)):
        ap = np.mean(ps_curve[k])
        ax.plot(
            rs,
            ps_curve[k],
            color=cs[k],
            label=str(f'[{ap:.3f}]' + types[k]),
            linewidth=0.5)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.title(figure_title)
    plt.legend()
    # plt.show()
    img = fig2im(fig)
    vis.add_image(f'{figure_title}.png', img)
    plt.close(fig)
