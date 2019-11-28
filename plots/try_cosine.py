import numpy as np
import torch
import matplotlib.patches as patches
from matplotlib import pyplot as plt

if __name__ == '__main__':
    
    
    
    # cls_scores
    upper_limit = 2
    lower_limit = -1
    split = 0.001
    cls_scores = np.arange(lower_limit, upper_limit, split)
    gammas = [0, 2, 2.5, 3, 5]
    # losses 
    ax = plt.subplot(121)
    
    for gamma in gammas:
        loss_cos = np.power(1-cls_scores, gamma)*(np.cos((1.57)*cls_scores+1.57)+1)
        label = "Cos loss w/gamma={}".format(gamma)
        ax.plot(cls_scores, loss_cos, label = label, linewidth=2.0)
    
    loss_CE = -1*np.log(cls_scores) 
    ax.plot(cls_scores, loss_CE, label="CE", linewidth=2.0)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(-1.75, 2)
    ax.set_xlim(-0.25, 1.5)
    ax.grid()
    ax.legend()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$Loss$')
    plt.title("Losses")
    plt.tight_layout()

    # Derivatives
    ax = plt.subplot(122)
    for gamma in gammas:
        der_cos = ((-1*gamma*np.power(1-cls_scores, gamma-1))*(np.cos(1.57*cls_scores+1.57)+1)) \
                  + (np.power(1-cls_scores, gamma)*(-1.57*np.sin(1.57*cls_scores+1.57)))
        label = "Cos loss w/gamma={}".format(gamma)
        ax.plot(cls_scores, der_cos, label=label, linewidth=2.0)

    derivative_CE = -1 / cls_scores
    ax.plot(cls_scores, derivative_CE, label="CE", linewidth=2.0)
    plt.title("Derivatives")
    plt.tight_layout()
    # Derivatives
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    
    ax.set_ylim(-10, 2)
    ax.set_xlim(-0.25, 1.5)
    
    ax.legend()
    ax.grid()
    plt.xlabel(r'$GT probability$')
    plt.ylabel(r'$\Delta Loss$')
    plt.tight_layout()

    # show
    plt.show()
    

