import torch
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from textwrap import fill
def ave_ratio_s(ColourDM,MotionDM,qua):
    dhc=[]
    dvc=[]
    dhm=[]
    dvm=[]
    set=[]
    for comb1 in qua:
        for comb2 in qua:
            if comb1!=comb2:
                idx1 = points[comb1] #[0] color, [1] motion
                idx2 = points[comb2]
                if idx1[0] == idx2[0] : # same color but differ motion
                    dhc.append(float(ColourDM[comb1-1,comb2-1]))
                    dhm.append(float(MotionDM[comb1-1,comb2-1]))
                    # c0+=1
                elif idx1[1] == idx2[1] : # same motion but differ color
                    dvc.append(float(ColourDM[comb1-1,comb2-1]))
                    dvm.append(float(MotionDM[comb1-1,comb2-1]))

    return dhc,dvc,dhm,dvm


def get_distance_from(x,y,task):
    distance_m = []
    distance_c=[]
    for i in range(len(x)):
        point0 = x[i]
        point1 = y[i]

        if task=='color' and i<len(x)/2:
            distance_m.append(point0)
            distance_c.append(point1)
        elif task =='motion' and i>=len(x)/2:
            distance_m.append(point0)
            distance_c.append(point1)
    return np.mean(distance_m), np.mean(distance_c)


def plot_compare(x,y,itv=1):
    name = ['match on color' for i in range(int(len(x) / 2))] + ['match on motion' for i in range(int(len(y) / 2))]
    data = {'dc': x, 'dm': y, 'Set': name}
    dhcdhm = pd.DataFrame(data)

    sns.set(style='white',font_scale=1.5) 
    et=max(max(x),max(x)) + itv
    st=min(min(y),min(y)) - itv
    g = sns.jointplot(x="dc", y="dm", hue="Set", data=dhcdhm,height=6,xlim=(st,et),ylim=(st,et))

    g.ax_joint.set_xlabel("Distance in motion task",fontsize=24)
    g.ax_joint.set_ylabel("Distance in color task",fontsize=24)
    g.ax_joint.axline((0, 0), slope=1,c='k')

    plt.legend(fontsize=17)
    plt.show()



if __name__=="__main__":

    parent_save_dir = '/home/user/../data'
    save_dir = parent_save_dir + '/save_dir'
    data_dir = '/home/user/data'
    upper_time = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22]

    np.set_printoptions(precision=2)

    points={1:[90, 90],2:[90, 30],3:[90, -30],4:[90, -90],
        5:[30, 90],6:[30, 30],7:[30, -30],8:[30, -90],
        9:[-30, 90],10:[-30, 30],11:[-30, -30],12:[-30, -90],
        13:[-90, 90],14:[-90, 30],15:[-90, -30],16:[-90, -90]}

    I=[11,12,15,16]
    II=[9,10,13,14]
    III=[1,2,5,6]
    IV=[3,4,7,8]

    tim = []
    dis = []
    q = []

    # 2D plot
    ISI_dhc = []
    ISI_dvc = []
    ISI_dhm = []
    ISI_dvm = []
    MotionDM = torch.load(save_dir+f'/Motion_ISI.pt') # or LSTM representations
    ColourDM = torch.load(save_dir+f'/Color_ISI.pt')
    dhc, dvc, dhm, dvm = ave_ratio_s(ColourDM, MotionDM, I + II + III + IV)
    ISI_dhc = ISI_dhc + dhc
    ISI_dhm = ISI_dhm + dhm
    ISI_dvc = ISI_dvc + dvc
    ISI_dvm = ISI_dvm + dvm
    font_size = 2
    y = ISI_dhc + ISI_dvc
    x = ISI_dhm + ISI_dvm
    plot_compare(x, y)


    # line plot
    method = 'ISI'
    for task in ['color','motion']:
        tim = []
        dis_m = []
        dis_c= []
        q_m = []
        q_c=[]
        for time in upper_time:
            MotionDM = torch.load(save_dir+f'motion_{method}_{time}.pt')
            ColourDM = torch.load(save_dir+f'color_{method}_{time}.pt')
            dhc,dvc,dhm,dvm = ave_ratio_s(ColourDM,MotionDM,I+II+III+IV)
            ISI_dhc = dhc
            ISI_dhm = dhm
            ISI_dvc = dvc
            ISI_dvm = dvm
            font_size=1.3
            y=ISI_dhc+ISI_dvc
            x=ISI_dhm+ISI_dvm
            m,c = get_distance_from(x,y,task)
            dis_m.append(m)
            dis_c.append(c)
            tim.append(time)
        plt.plot(tim,dis_m,'s-')
        plt.plot(tim,dis_c,'^-')
        if task == 'color':
            mat_color_under_m = dis_m
            mat_color_under_c = dis_c
        elif task == 'motion':
            mat_motion_under_m = dis_m
            mat_motion_under_c = dis_c

    print(f't test: {stats.ttest_rel(mat_color_under_c, mat_color_under_m)}')  # D_m
    print(f't test: {stats.ttest_rel(mat_motion_under_m, mat_motion_under_c)}')  #D_c


    labels = [r'$D_{m}(i,j)^{c}$',r'$D_{c}(i,j)^c$',r'$D_{m}(i,j)^m$',r'$D_{c}(i,j)^m$']
    plt.legend(labels, bbox_to_anchor=(1, 0.5),fontsize=20)
    plt.xlabel("Time (second)", fontsize=20)
    plt.ylabel("Distance", fontsize=20)
    plt.xticks( fontsize = 20 )
    plt.yticks( fontsize=20 )
    plt.show() 

