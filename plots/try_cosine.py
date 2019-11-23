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
    loss_cos_neg = -1*np.cos(((1.57)*cls_scores+1.57))
    gammas = [0, 1, 1.5, 2, 5, 10] 
    ax = plt.subplot(121)
    for gamma in gammas:
        loss_cos_pos = np.power(1-cls_scores, gamma)*(np.cos((1.57)*cls_scores+1.57)+1)
        label = "Cos loss w/gamma={}".format(gamma)
        ax.plot(cls_scores, loss_cos_pos, label = label)
    loss_cos_nf = np.cos(1.57*cls_scores)
    loss_linear = -1*(cls_scores-1)
    loss_CE_pos = -1*np.log(cls_scores)
    loss_CE_neg = -1*np.log(1-cls_scores)
    loss_linear_pos = (-1*cls_scores)+1
    loss_linear_neg = (cls_scores)
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
    # Derivatives
    ax = plt.subplot(121)
    derivative_cos = -1.57*np.sin(1.57*cls_scores+1.57)
    derivative_cos_nf = -1.57*np.sin(1.57*cls_scores)
    derivative_linear_pos = -1*np.ones(3*1000)
    derivative_linear_neg = 1*np.ones(3*1000)
    derivative_CE = -1 / cls_scores
    # Losses
    #ax.plot(cls_scores, loss_cos_pos, label="Cosine Positive")
    #ax.plot(cls_scores, loss_cos_neg, label="Cosine Negative")
    ax.plot(cls_scores, loss_CE_pos, label="CE")
    #ax.plot(cls_scores, loss_CE_neg, label="CE_neg")
    #ax.plot(cls_scores, loss_linear_pos, label = "linear_pos")
    #ax.plot(cls_scores, loss_linear_neg, label = "linear_neg")
    #ax.add_patch(rect_losses)
    #ax.add_patch(rect_derivatives)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(-1.75, 2)
    ax.set_xlim(-0.25, 1.5)
    #ax.plot(cls_scores, loss_linear, label="Linear")
    ax.legend()
    ax.grid()
    plt.xlabel(r'$CLS probs$')
    plt.ylabel(r'$Loss$')
    plt.title("Losses")
    plt.tight_layout()
    ax = plt.subplot(122)
    #ax.plot(cls_scores, derivative_cos, label="Cosine Flipped")
    #ax.plot(cls_scores, derivative_cos_nf, label="Cosine Non Flipped")
    #ax.plot(cls_scores, derivative_CE, label="CE Derivative")
    #ax.plot(cls_scores, derivative_linear_pos, label = "linear_pos")
    #ax.plot(cls_scores, derivative_linear_neg, label = "linear_neg")
    plt.title("Derivatives")
    plt.tight_layout()
    # Derivatives
    #ax.plot(cls_scores, derivative_cos, label="Derivative Cosine")
    #ax.plot(cls_scores, derivative_linear, label="Derivative Linear")
    #ax.plot(cls_scores, derivative_CE, label="Derivative CE")
    #ax.add_patch(rect_losses)
    #ax.add_patch(rect_derivatives)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    
    ax.set_ylim(-1.75, 2)
    ax.set_xlim(-0.25, 1.5)
    
    ax.legend()
    ax.grid()
    plt.xlabel(r'$CLS probs$')
    plt.ylabel(r'$\Delta Loss$')
    plt.show()
    

