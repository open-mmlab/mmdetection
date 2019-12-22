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

    #gammas = [3, 4, 5, 10]
    #lw_vals = [2.5, 3.1, 5, 10]
    
    if LW_SEARCH == True:
        gamma = 2
        gamma_lw = []
        for lw in range(1, 21, 2):
            param_sub = []
            param_sub = [gamma, lw]
            gamma_lw.append(param_sub)
    else:    
        gamma_lw = [[10, 1], \
        [10, 2], \
        [10, 2.5], \
        [10, 3], \
        [10, 4],  \
        [10, 5], \
        [10, 6], \
        [10, 7], \
        [10, 8]]
        
        #gamma_lw = [[2, 1.5], \
        #[2, 2.0], \
        #[2, 2.5],  \
        #[2, 3.0], \
        #[2, 4.0], \
        #[2, 5.0]] 
    # losses 
    ax = plt.subplot(121)
    plt.plot(figsize=(20,20))
    #for gamma in gammas:
    #    loss_cos = np.power(1-cls_scores, gamma)*(np.cos((1.57)*cls_scores+1.57)+1)
    #    if lw_vals != None:
    #        linewidth = 2
    #        for lw in lw_vals:
    #            loss_cos = loss_cos * lw
    #            label = "Cos loss w/gamma={}, lw={}".format(gamma, lw)
    #            ax.plot(cls_scores, loss_cos, label = label, linewidth=linewidth)
    #            linewidth += 0.7
    loss_CE = -1*np.log(cls_scores) 
    print("LOSS SIMILARITY") 
    for param in gamma_lw:
        gamma = param[0]; lw = param[1]
        loss_cos = np.power(1-cls_scores, gamma)*(np.cos((1.57)*cls_scores+1.57)+1)
        loss_cos = loss_cos * lw
        label = "Cos Loss w/gamma={}, lw={}".format(gamma,lw)
        corr = np.correlate(loss_CE, loss_cos)[0]
        print("gamma:{}, lw:{} => corr:{}".format(gamma,lw,corr))
        ax.plot(cls_scores, loss_cos, label = label, linewidth = 2.0)
 
    ax.plot(cls_scores, loss_CE, label="CE", linewidth=5.0)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(-1.75, 20)
    ax.set_xlim(-0.25, 1.5)
    ax.grid()
    legend = ax.legend()
    export_legend(legend, "losses_legend.png")
    #ax.get_legend().remove()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$Loss$')
    plt.title("Losses")
    plt.tight_layout()

    derivative_CE = -1 / cls_scores
    # Derivatives
    ax = plt.subplot(122)
    #for gamma in gammas:
    #    der_cos = ((-1*gamma*np.power(1-cls_scores, gamma-1))*(np.cos(1.57*cls_scores+1.57)+1)) \
    #              + (np.power(1-cls_scores, gamma)*(-1.57*np.sin(1.57*cls_scores+1.57)))
    #    if lw_vals != None:
    #        linewidth=2
    #        for lw in lw_vals:
    #            der_cos = der_cos * lw
    #            label = "Cos loss w/gamma={}, lw={}".format(gamma, lw)
    #            corr = np.correlate(derivative_CE, der_cos)[0]
    #            print("gamma:{}, lw:{} => corr:{}".format(gamma, lw, corr))
    #            ax.plot(cls_scores, der_cos, label=label, linewidth=linewidth)
    #            linewidth += 0.7 
    print("DERIVATICES SIMILARITY") 
    for param in gamma_lw:
        gamma=param[0]; lw=param[1]
        der_cos = ((-1*gamma*np.power(1-cls_scores, gamma-1))*(np.cos(1.57*cls_scores+1.57)+1)) \
                  + (np.power(1-cls_scores, gamma)*(-1.57*np.sin(1.57*cls_scores+1.57)))
        der_cos = der_cos * lw
        label = "Cos loss w/gamma={}, lw={}".format(gamma,lw)
        corr = np.correlate(derivative_CE, der_cos)[0]
        print("gamma:{}, lw:{} => corr:{}".format(gamma, lw, corr))
        ax.plot(cls_scores, der_cos, label=label, linewidth = 2.0)

    ax.plot(cls_scores, derivative_CE, label="CE", linewidth=5.0)
    plt.title("Derivatives")
    plt.tight_layout()
    # Derivatives
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    
    ax.set_ylim(-30, 2)
    ax.set_xlim(-0.25, 1.5)
    
    legend = ax.legend()
    export_legend(legend, "derivatives_legend.png")
    #ax.get_legend().remove()
    ax.grid()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$\Delta Loss$')
    plt.tight_layout()

    # show
    plt.show()
    

