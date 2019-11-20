import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # cls_scores
    upper_limit = 2
    lower_limit = -1
    split = 0.001
    cls_scores = np.arange(lower_limit, upper_limit, split)
    # losses
    loss_cos = np.cos(1.57*cls_scores)
    loss_linear = -1*(cls_scores-1)
    # Plots
    ax = plt.subplot(111)
    derivative_cos = -1.57*np.sin(1.57*cls_scores)
    derivative_linear = -1*np.ones(3*1000)
    # Losses
    ax.plot(cls_scores, loss_cos, label="Cosine")
    ax.plot(cls_scores, loss_linear, label="Linear")
    # Derivatives
    ax.plot(cls_scores, derivative_cos, label="Derivative Cosine")
    ax.plot(cls_scores, derivative_linear, label="Derivative Linear")
    
    # Rectangles
    rect_losses = patches.Rectangle((0,0), 1, 1,\
                             linewidth=2,\
                             linestyle='--',\
                             edgecolor='#B9D8EC',\
                             facecolor='#EFE7DE')
    
    rect_derivatives = patches.Rectangle((0,0), 1, -2,\
                                         linewidth=2,\
                                         linestyle='--',\
                                         edgecolor='#B9D8EC',\
                                         facecolor='#EFE7DE')
    ax.add_patch(rect_losses)
    ax.add_patch(rect_derivatives)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    
    ax.set_ylim(-1.75, 2)
    ax.set_xlim(-1,1.25)
    
    ax.legend()
    ax.grid()
    plt.xlabel(r'$CLS probs$')
    plt.ylabel(r'$\Delta Loss, Loss$')
    plt.show()
    

