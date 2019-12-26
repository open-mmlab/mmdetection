import numpy as np
import torch
import matplotlib.patches as patches
from matplotlib import pyplot as plt

LW_SEARCH = False

def export_legend(legend, filename):
    fig = legend.figure
    fig.canvas.draw()
    bbox=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == '__main__':
    
    # cls_scores
    upper_limit = 2
    lower_limit = -1
    split = 0.001
    eps = 1e-6
    cls_scores = np.arange(lower_limit, upper_limit, split)
    
    scaler_vals = [0.1, 0.2, 0.3]
    scaler_vals = np.asarray(scaler_vals)

    # losses.
    ax = plt.subplot(121) 
    loss_CE = -1*np.log(cls_scores) 
    for scaler_val in scaler_vals:
        loss_exp = np.exp((-cls_scores)/scaler_val)
        label = "EXP w/ scaler={}".format(scaler_val)
        ax.plot(cls_scores, loss_exp, label=label, linewidth=3.0) 
    ax.plot(cls_scores, loss_CE, label="CE", linewidth=3.0)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(-1.75, 2)
    ax.set_xlim(-0.25, 1.5)
    ax.grid()
    legend = ax.legend()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$Loss$')
    plt.title("Losses")
    plt.tight_layout()
    
    # derivatives.
    ax = plt.subplot(122)  
    derivative_CE = -1 / cls_scores
    for scaler_val in scaler_vals:
        derivative_EXP = (-1./scaler_val)*np.exp((-cls_scores)/scaler_val)
        label = "EXP w/ scaler={}".format(scaler_val)
        ax.plot(cls_scores, derivative_EXP, label=label, linewidth=3.0)
    
    ax.plot(cls_scores, derivative_CE, label="CE", linewidth=3.0)
    plt.title("Derivatives")
    plt.tight_layout()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    
    ax.set_ylim(-10, 2)
    ax.set_xlim(-0.25, 1.5)
    
    legend = ax.legend()
    ax.grid()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$\Delta Loss$')
    plt.tight_layout()

    # show
    plt.show()
    

