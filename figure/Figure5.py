# spearman correlation with time window
import pandas as pd
import os
# os.chdir(os.path.dirname(__file__))
from scipy import stats
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import itertools
import matplotlib

def compare(Mtr, Mat):
    Mcolour_vec = torch.zeros(120)
    Base_vec = np.zeros(120)
    cc = 0
    for row in range(16):
        for col in range(row + 1, 16):
            Mcolour_vec[cc] = Mat[row, col]
            Base_vec[cc] = Mtr[row, col]
            cc += 1

    [corr, pvalue] = stats.spearmanr(Base_vec, Mcolour_vec)
    return corr


points = {'[0, 0]': 0, '[-5, 0]': 1, '[5, 0]': 2, '[0, -5]': 3, '[0, 5]': 4, '[90, 90]': 5, '[90, 30]': 6,
          '[90, -30]': 7, '[90, -90]': 8,
          '[30, 90]': 9, '[30, 30]': 10, '[30, -30]': 11, '[30, -90]': 12, '[-30, 90]': 13, '[-30, 30]': 14,
          '[-30, -30]': 15,
          '[-30, -90]': 16, '[-90, 90]': 17, '[-90, 30]': 18, '[-90, -30]': 19, '[-90, -90]': 20}

parent_save_dir = '/home/user/../data'
save_dir = parent_save_dir + '/save_dir'
data_dir = '/home/user/data'
data=pd.read_csv(parent_save_dir + f'data_sample.csv')
data = data.dropna()
data_mat = np.asarray(data)

for baseline in ['distance']: 
    color = ['#d53e4f',
            '#fc8d59',
            '#fee08b',
            '#e6f598',
            '#99d594',
            '#3288bd']
    for time in range(90,150+1):
        time_c = time / 60 - 1.5
        k = 0
        plt.figure(figsize=(10, 20))
        for task in ['motion', 'color']:
            H= torch.zeros(6,6) # row is region, column is layer
            for roi in ['PFC', 'FEF', 'LIP', 'IT', 'MT', 'V4']:
                if task == 'motion':
                    baseline_matrix = torch.load(f'{roi}_Motion_ISI.pt')
                elif task == 'color':
                    baseline_matrix = torch.load(f'{roi}_Color_ISI.pt')
                for layer in [1,2,3,4,5,6]:
                    correlation = []
                    test_matrix = torch.load(f'/home/user/.../{layer}layer_{task}_{time}.pt')

                    H[['PFC', 'FEF', 'LIP', 'IT', 'MT', 'V4'].index(roi), layer - 1] = compare(test_matrix,
                                                                                               baseline_matrix)
            CM = H.numpy()
            cmap = plt.get_cmap('Purples')
            norm = matplotlib.colors.Normalize(vmin=-0.2,vmax=0.6)
            if task=='motion':
                plt.subplot(211)
                plt.imshow(CM, interpolation='nearest', cmap=cmap,norm=norm)
                plt.title('Motion task',fontsize=25)
            elif task=='color':
                plt.subplot(212)
                plt.imshow(CM, interpolation='nearest', cmap=cmap,norm=norm)
                plt.title('Color task',fontsize=30)

            roi_names = ['PFC', 'FEF', 'LIP', 'IT', 'MT', 'V4']
            layer_names = ['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6']
            tick_marks = np.arange(len(roi_names))
            plt.xticks(tick_marks, layer_names, rotation=30,fontsize=30)
            plt.yticks(tick_marks, roi_names,fontsize=30)
            plt.tick_params(axis='both', 
                            direction='out', 
                            length=10, 
                            width=2, 
                            pad=10)
            cbar = plt.colorbar(ticks=([-0.2, 0.6]),fraction=0.03)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(30)
            thresh = CM.mean()
            for i, j in itertools.product(range(CM.shape[0]), range(CM.shape[1])):
                plt.text(j, i, "{:.1f}".format(CM[i, j]),fontsize=30,
                         horizontalalignment="center",
                         color="white" if CM[i,j]>=0.2 else "black")
        plt.tight_layout(pad=1)
        plt.close()
