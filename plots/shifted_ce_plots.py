import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pdb

def focal(cls_scores, gamma, is_shifted=True):
    # focal loss
    if is_shifted:
    # if shifted version.
        return (torch.pow(1-cls_scores, exponent=gamma)) * \
                -torch.log(cls_scores + \
                torch.exp(torch.tensor(-1, dtype=torch.float))) # shifting factor
    else:
    # original cross entropy
        return (torch.pow(1-cls_scores, exponent=gamma)) * \
                -torch.log(cls_scores) # no shift factor

def focal_der(cls_scores, gamma, is_shifted = True):
    # focal loss derivative
    if is_shifted:
    # if shifted version.
        if gamma == 0:
            # shifted CE
            return -1*(1/(cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))))
        else:
            # shifted FocalLoss
            return torch.pow((1-cls_scores), exponent=gamma-1) * \
                   (gamma*torch.log(cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) * \
                   (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) + \
                   (cls_scores-1))/ \
                   (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float)))
                   
    else:
    # original cross entropy
         return -1 / cls_scores
         
def main(cls_scores, gammas, is_shifted, legend_text):
    print("Running Losses.")
    print("-----------------------------------------------")
    # Plot Loss Values
    ax = plt.subplot(111)
    legend_it = 0
    for gamma in gammas:
        focal_val = focal(cls_scores, \
                          gamma, \
                          is_shifted[legend_it])
        # find y-axis intersection.
        idx = (cls_scores>=0).nonzero()[0]
        # find x-axis intersection.
        if (focal_val>=0).nonzero().size()[0] == 0:
            val = 'No intersection'
        else:
            idx = (focal_val<0).nonzero()[0]
            val = cls_scores[idx].numpy()[0]
        print("{} Intersects X Axis at: {:.4f}, Y Axis at: {:.4f}"\
              .format(legend_text[legend_it], \
                      val, focal_val[idx].numpy()[0]))
        ax.plot(cls_scores, focal_val,\
                label=legend_text[legend_it],linewidth=2)
        legend_it += 1
    print("------------------------------------------------")
    # figure properties
    ax.set_ylim(-0.6,1.6)
    ax.set_xlim(-0.25, 1.5)
    rect=patches.Rectangle((0, -0.1), 1, 1.1,\
                           linewidth=2,\
                           linestyle='--',\
                           edgecolor='#B9D8EC',\
                           facecolor='#EFE7DE')
    # 1 intersection for CE.
    #ax.plot(torch.exp(torch.tensor(-1, dtype=torch.float)), -torch.log(torch.exp(torch.tensor(-1, dtype=torch.float))), 'r', marker='o', markersize=12)
    ax.add_patch(rect)
    plt.legend()
    plt.xlabel(r'$Cls prob$')
    plt.ylabel(r'$Loss$')
    ax.grid(linestyle='-', linewidth=0.5)
    plt.show()
    
    print("Running Derivatives.")
    print("------------------------------------------------")
    # Plot Derivatives
    ax = plt.subplot(111)
    legend_it = 0
    for gamma in gammas:
        focal_val = focal_der(cls_scores, \
                              gamma, \
                              is_shifted[legend_it])
        # find y-axis intersection.
        idx = (cls_scores>=0).nonzero()[0]
        # find x-axis intersection.
        if (focal_val>=0).nonzero().size()[0] == 0:
            val = 'No intersection'
        else:
            idx = (focal_val>=0).nonzero()[0]
            val = cls_scores[idx].numpy()[0]
        print("{} Intersects X Axis at: {:.4f}, Y Axis at: {:.4f}"\
              .format(legend_text[legend_it],\
              val, focal_val[idx].numpy()[0]))
        ax.plot(cls_scores, focal_val,\
                label=legend_text[legend_it],linewidth=2)
        legend_it += 1
    print("-----------------------------------------------")
    # figure properties
    ax.set_ylim(-10,3)
    ax.set_xlim(-0.25,1.5)
    rect=patches.Rectangle((0, 0), 10.0, -10,\
                           linewidth=2,\
                           linestyle='--',\
                           edgecolor='#B9D8EC',\
                           facecolor='#EFE7DE')
    ax.add_patch(rect)
    #plt.title('CE vs. Shifted CE')
    plt.legend()
    plt.xlabel(r'$Cls prob$')
    plt.ylabel(r'$\Delta Loss$')
    ax.grid(linestyle='-', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    # get classification scores (x-axis)
    cls_scores = np.arange(-2, 2, 0.01)
    # gamma set.   
    #gammas = [0, 0, 1, 2, 4, 6, 8, 10]
    #is_shifted = [False, True, True, True, True, True, True, True]
    #legend_text = ['CE', 'Shifted-CE', 'gamma=1', 'gamma=2', 'gamma=4', 'gamma=6', 'gamma=8', 'gamma=10']
    #main(torch.tensor(cls_scores), gammas, is_shifted, legend_text)
    
    gammas = [0, 0]
    is_shifted = [False, True]
    legend_text = ['CE', 'Shifted-CE']
    
    main(torch.tensor(cls_scores), gammas, is_shifted, legend_text)

# ---FIRST DERIVATION--- #

#return -1*torch.pow((1-cls_scores), exponent=gamma-1) * \
#                   (torch.log(cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) * \
#                   (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) + \
#                   torch.pow((1-cls_scores), exponent=gamma+1))/      (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float)))

# ---SECOND DERIVATION--- #

#return torch.pow((1-cls_scores), exponent=gamma-1) * \
#                   (gamma*torch.log(cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) * \
#                   (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float))) + \
#                   (cls_scores-1))/ \
#                   (cls_scores+torch.exp(torch.tensor(-1, dtype=torch.float)))
