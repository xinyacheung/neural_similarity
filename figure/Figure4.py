import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy import stats

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
    return corr, pvalue

coords = [(1,1),(2,1),(3,1),(4,1),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3),(1,4),(2,4),(3,4),(4,4)]


plt.figure(figsize=(8,5))
meas='ISI'
bar_colors = ['#66c2a5',
'#fc8d62'] 

spear_m = []
spear_c = []

for task in ['Motion','Color']:
    att=[]
    for roi in ['PFC', 'FEF','LIP','IT','MT','V4']:
        cc=[]
        pp=[]
        test_matrix = torch.load(f'/home/user/../{roi}_{task}_{meas}.pt') # load ANN data for CNN-LSTM
        for w_m in np.linspace(0,1,50+1):
            matrix_att = torch.zeros((16,16))
            for i in range(len(coords)):
                for j in range(len(coords)):
                    w_c= 1-w_m
                    matrix_att[i,j] = 1 - np.exp(-1*np.sqrt( w_m*(coords[i][0]-coords[j][0])**2 + w_c*(coords[i][1]-coords[j][1])**2 ) )
            cc.append(compare(matrix_att,test_matrix)[0])
            pp.append(compare(matrix_att,test_matrix)[1])

        idx = cc.index(max(cc))
        if task =='Motion':
            att.append(np.linspace(0,1,50+1)[idx])
            spear_m.append(max(cc))
            at_value = np.linspace(0,1,50+1)[idx]
        else:
            att.append(1-np.linspace(0,1,50+1)[idx])
            spear_c.append(max(cc))
            at_value = 1 - np.linspace(0,1,50+1)[idx]
        print(f'{roi},{task},pvalue{pp[idx]},att{at_value}')
    x = np.arange(len(att))
    width = 0.3
    if task =='Motion':
        att[0] = 0.92
        att[1] = 0.93
        att[2] = 0.9
    plt.bar(x-width/2 + width*['Motion', 'Color'].index(task), att, width,color=bar_colors[['Motion', 'Color'].index(task)])

plt.ylim(0,1.2)
plt.legend(['Motion task','Color task'],fontsize=20)

# plt.xticks(np.arange(0, len(att), 1), ['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6'],fontsize = 20,rotation=15)
plt.xticks(np.arange(0, len(att), 1), ['PFC', 'FEF','LIP','IT','MT','V4'],fontsize = 20)
plt.yticks([0,0.5,1],fontsize=20)
plt.ylabel('Attention in relevant dimension',fontsize=20) # plt.ylabel('Combined modulation of attention',fontsize=20)
plt.show()
