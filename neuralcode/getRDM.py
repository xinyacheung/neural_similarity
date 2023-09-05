'''
This file calculates measure-based RDM.
'''

import glob
import numpy as np
from scipy import stats
import re
import pandas as pd
import torch

parent_save_dir = '/home/user/../data'
save_dir = parent_save_dir + '/save_dir'
data_dir = '/home/user/data'


# get ISI & SPIKE-based RDM at each time bin
mat_files_match = '*.csv'
data_files = glob.glob(mat_files_match)
data_files.sort()
data = []
for i in data_files:
    if i != 'data.csv':
        csv_data = pd.read_csv(i, sep=',')
        data.append(csv_data)
    
data = np.vstack(data)
data_pd = pd.DataFrame(data, columns=csv_data.columns)
data_pd.to_csv(parent_save_dir + 'data.csv')

data=pd.read_csv(parent_save_dir + f'data_sample.csv')
data=data.dropna()
data_mat=np.asarray(data)

upper_time = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21]
points={'[0, 0]':0,'[-5, 0]':1,'[5, 0]':2,'[0, -5]':3,'[0, 5]':4,'[90, 90]':5,'[90, 30]':6,'[90, -30]':7,'[90, -90]':8,
        '[30, 90]':9,'[30, 30]':10,'[30, -30]':11,'[30, -90]':12,'[-30, 90]':13,'[-30, 30]':14,'[-30, -30]':15,
        '[-30, -90]':16,'[-90, 90]':17,'[-90, 30]':18,'[-90, -30]':19,'[-90, -90]':20}

for task in ['motion','color']:
        data_co=data_mat[np.where(data_mat[:,5]==task),:]
        data_co=np.squeeze(data_co,0) 
        matrix=data_co
        for method in ['ISI','SPIKE']:
            for time in upper_time:
                n,m=matrix.shape
                x1=matrix[:,9]
                y1=matrix[:,10]
                x2=matrix[:,11]
                y2=matrix[:,12]
                
                if method =='ISI':
                    dist=matrix[:,16+upper_time.index(time)*3] 
                if method =='SPIKE':
                    dist=matrix[:,17+upper_time.index(time)*3] 
                
                ED=torch.zeros(16,16)
                counts=torch.ones(16,16)
                for i in range(n):
                    point1=str([x1[i],y1[i]])
                    point2=str([x2[i],y2[i]])
                    key_row=points[point1]
                    key_col=points[point2]
                
                    if (key_row > 4) and (key_col > 4):
                        count=counts[key_row-5,key_col-5]
                        ED[key_row-5,key_col-5] +=dist[i]
                        ED[key_col-5,key_row-5] += dist[i]
                        counts[key_row-5,key_col-5] += 1
                        counts[key_col-5,key_row-5] +=1
                
                ED=ED/counts
                torch.save(ED, save_dir+f'{task}_{method}_{time}.pt')


# get ISI & SPIKE-based RDM
data=pd.read_csv(parent_save_dir + f'data_sample.csv')
data=data.dropna(subset=['isi_full']) #(subset=['spike_full'])
data_mat=np.asarray(data)

#Condition: Colour
data_col=data_mat[np.where(data_mat[:,5]=='color'),:]
data_col=np.squeeze(data_col,0)
print(data_col.shape)
#Condition: Motion
data_mot=data_mat[np.where(data_mat[:,5]=='motion'),:]
data_mot=np.squeeze(data_mot,0)
print(data_mot.shape)

matrix=data_col # data_mot
n,m=matrix.shape
x1=matrix[:,9]
y1=matrix[:,10]
x2=matrix[:,11]
y2=matrix[:,12]
dist=matrix[:,13]
EDcolour=torch.zeros(16,16)
counts=torch.ones(16,16)

for i in range(n):
    point1=str([x1[i],y1[i]])
    point2=str([x2[i],y2[i]])
    key_row=points[point1]
    key_col=points[point2]
    #print(key_row,key_col)
    if (key_row > 4) and (key_col > 4):
        count=counts[key_row-5,key_col-5]
        EDcolour[key_row-5,key_col-5] +=dist[i]
        EDcolour[key_col-5,key_row-5] += dist[i]
        counts[key_row-5,key_col-5] += 1
        counts[key_col-5,key_row-5] +=1

EDcolour=EDcolour/counts
torch.save(EDcolour,save_dir+'Color_ISI.pt')


# get RDM based on Pearson or Cosine etc. distance measure at each time bin
mat_files_match = data_dir + '/*.mat'
data_files = glob.glob(mat_files_match)
data_files.sort()
sess=[]
for file_name_i in data_files:
    try:
        sess_num = re.search(r'(/)([0-9]*)(.mat)', file_name_i).group(2)
    except:
        sess_num = re.search(r'(/)([0-9]*_[0-9]*)(.mat)', file_name_i).group(2)
    sess.append(sess_num)
    
upper_time2 = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21]

measure = 'Pearson'
for task in ['color', 'motion']:
    for type in ['lobe','roi']:
        for t1 in upper_time2:
            if type =='lobe':
                temp = np.zeros((16,3))
                for session in sess:
                    temp = (temp+np.load(save_dir+f'/{task}_{session}_{t1}_lobe.npy'))/2.0
            if type == 'roi':
                temp = np.zeros((16,6))
                for session in sess:
                    temp = (temp+np.load(save_dir+f'/{task}_{session}_{t1}_roi.npy'))/2.0

            matrix = torch.zeros(16,16)
            for i in range(16):
                for j in range(16):
                    matrix[i,j] = stats.pearsonr(temp[i], temp[j])[0] # other measures etc.
            torch.save(matrix, save_dir + f'/{measure}({type}_{t1}).pt')