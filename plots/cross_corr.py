import numpy as np
import matplotlib.pyplot as plt
import pdb


def cos_loss(cls_scores, loss_weight, gamma):
    '''
    Derivative of cos_loss, controlled by loss weight
    and gamma.
    '''
    return  loss_weight * (((-1*gamma*np.power(1-cls_scores, gamma-1))*(np.cos(1.57*cls_scores+1.57)+1)) \
            + (np.power(1-cls_scores, gamma)*(-1.57*np.sin(1.57*cls_scores+1.57))))

def cross_entropy(cls_scores):
    '''
    Derivative of cross entropy
    '''
    return -1/cls_scores

def main(plot=False):
    floor=-1; step = 0.001; ceil=2
    cls_scores = np.arange(floor,ceil+step,step)
    
    loss_weight=3.1
    gamma=4
   
    cos_vals = cos_loss(cls_scores, loss_weight, gamma)
    ce_vals = cross_entropy(cls_scores)

    if plot == True:
        plt.plot(cls_scores, cos_vals)
        plt.plot(cls_scores, ce_vals)
        plt.ylim([-30, 2])
        plt.xlim([-0.25,1.5])
        plt.gca().axhline(0 ,color='black')
        plt.gca().axvline(0, color='black')
        plt.grid()
        plt.show()

    lws = [5, 10]
    gammas = [3, 4]
    
    corr_scores = []
    axs = []
    for lw in lws:
        for gamma in gammas:
            cos = cos_loss(cls_scores, lw, gamma)
            ce = cross_entropy(cls_scores)
            axs.append([gamma,lw])
            corr_score = np.correlate(ce, cos)
            corr_scores.append(corr_score[0])
            print("lw: {}, gamma: {}, corr_score:{}\n".format(lw, gamma, corr_score[0]))

if __name__ == '__main__':
    main()
