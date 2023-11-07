import pandas as pd
import os
from scipy import stats
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def compare(Mtr,Mat):              
    Mcolour_vec=torch.zeros(120)
    Base_vec=np.zeros(120)
    cc=0
    for row in range(16):
      for col in range(row+1,16):
        Mcolour_vec[cc]=Mat[row,col]
        Base_vec[cc]=Mtr[row,col]
        cc+=1
    [corr,pvalue] = stats.spearmanr(Base_vec,Mcolour_vec)
    return corr

parent_save_dir = '/home/user/../data'
save_dir = parent_save_dir + '/save_dir'

upper_time = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21]
upper_time2 = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21]
points={'[0, 0]':0,'[-5, 0]':1,'[5, 0]':2,'[0, -5]':3,'[0, 5]':4,'[90, 90]':5,'[90, 30]':6,'[90, -30]':7,'[90, -90]':8,
        '[30, 90]':9,'[30, 30]':10,'[30, -30]':11,'[30, -90]':12,'[-30, 90]':13,'[-30, 30]':14,'[-30, -30]':15,
        '[-30, -90]':16,'[-90, 90]':17,'[-90, 30]':18,'[-90, -30]':19,'[-90, -90]':20}

for baseline in ['distance']:
    if baseline == 'distance':
        baseline_matrix = torch.load(parent_save_dir+'stimuli_coordinates_matrix.pt')

    for task in ['motion','color']:

        plt.figure(figsize=(10,4))
        for method in ['ISI','SPIKE','Euclidean','Cosine','Pearson']:
            if method in ['ISI','SPIKE']:

                correlation = []
                for time in upper_time:

                    ED = torch.load(save_dir+f'{task}_{method}_{time}.pt')
                    correlation.append(compare(ED, baseline_matrix))
                plt.plot([i for i in upper_time],correlation,'-')

            else:

                correlation = []
                for time in upper_time2:
                    if method == 'Euclidean':
                        ED = torch.load(parent_save_dir+f'/Euclidean.pt')
                        correlation.append(compare(ED, baseline_matrix))
                    if method == 'Cosine':
                        ED = -1*torch.load(parent_save_dir+f'/Cosine.pt')
                        correlation.append(compare(ED, baseline_matrix))
                    if method == 'Pearson':
                        ED = -1*torch.load(parent_save_dir+f'/Pearson.pt')
                        correlation.append(compare(ED, baseline_matrix))
                plt.plot([i for i in upper_time], correlation, '-')

        if task =='motion':
            plt.legend( ['ISI distance','SPIKE distance','Euclidean','Cosine','Pearson'], bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0,fontsize=18)

        plt.xlabel('Time (second)',fontsize=20)
        plt.ylabel('Spearman correlation',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()






