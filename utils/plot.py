import os

import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
else:
    plt.switch_backend('tkagg')
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np

import torch
from torch.utils.data import *
import torchvision.transforms as transforms

# insert this before plt.show()
def max_window():
    backend = plt.get_backend()

    if backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        ### works on Ubuntu??? >> did NOT working on windows
        # mng.resize(*mng.window.maxsize())
        mng.window.state('zoomed') #works fine on Windows!
    elif backend == 'wxAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

def plot_batch(images):
    # image.shape = [n, c, h, w]

    # plot each sample in a col
    cols = images.shape[0]

    # plot each layer in a row
    rows = images.shape[1]

    images = images.cpu().numpy()

    plt.ion()
    f, axs = plt.subplots(rows, cols)

    for i in range(cols):
        for j in range(rows):
            image = images[i, j]
            ax = axs[j][i]

            im = ax.imshow(image, cmap='gray')
            f.colorbar(im, ax=ax, shrink=0.75)

            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.ioff()

    # max_window() 

    plt.show()
