from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import pdb


if __name__ == '__main__':
    p = Path('/home/cancam/workspace/mmdetection/class_analysis.txt')
    out_list = []
    with p.open('rb') as f:
        #fsz = os.fstat(f.fileno()).st_size
        out = np.loadtxt(f) 
    plt.hist([out[:, 0], out[:,2]])
    plt.show()
