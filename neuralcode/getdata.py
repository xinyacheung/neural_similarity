'''
This file alculates neural dissimilarity (by ISI & SPIKE) from spiking recordings. The spiking data from reference [Siegel, M., Buschman, T. J., & Miller, E. K. (2015). Cortical information flow during flexible sensorimotor decisions. Science, 348(6241), 1352-1355].
'''

import scipy.io
import glob
import pyspike as spk
import numpy as np
import re
import pandas as pd
import sys, os
from joblib import Parallel, delayed
import torch
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__

def scalar_dists(tmp_neuron, inds, t0, t1):
    try:
        isi_dist = spk.isi_distance(tmp_neuron, indices=inds, interval=(t0, t1))
    except:
        isi_dist = np.nan
    blockPrint() #for some reason spike distance is too verbose, so blocking output
    try:
        spike_dist = spk.spike_distance(tmp_neuron, indices=inds, interval=(t0, t1))
    except:
        spike_dist = np.nan
    enablePrint() #enable normal output to terminal again
    try:
        spike_sync = spk.spike_sync(tmp_neuron, indices=inds, interval=(t0, t1))
    except:
        spike_sync = np.nan
    return isi_dist, spike_dist, spike_sync

def compute_dists(spikes, labels, isolated_neurons, ref_dists, other_args, condition):
    unitInfo, monkey = other_args
    session = unitInfo.session[0]
    task = 'categorization'
    three_dists=[]

    for j in range(spikes.shape[1]): # neurons
        lobe = unitInfo.well[j]
        roi = unitInfo.area[j]
        if isolated_neurons[j]==0:
            continue                  #if neuron wasn't isolated then skip
        tmp_neuron=[]
        for i in range(spikes.shape[0]): # trials 
            if isinstance(spikes[i,j],float):
                tmp_neuron.append(spk.SpikeTrain([spikes[i,j]],edges))
            else:
                tmp_neuron.append(spk.SpikeTrain(spikes[i,j],edges))
        coords = np.array(ref_dists[['x1','y1','x2','y2']])
        for row_ind in range(len(coords)):
            row = coords[row_ind]
            print(f"{j}/{spikes.shape[1]}, {session}---{row_ind+1}/{len(coords)}")
            stim1_inds = np.where(np.array(labels.color==row[0]) & np.array(labels.direction==row[1])) #row[0] is x1, row[1] is y1
            stim2_inds = np.where(np.array(labels.color==row[2]) & np.array(labels.direction==row[3])) #row[2] is x2, row[3] is y2
            inds = np.sort(np.hstack([stim1_inds,stim2_inds]))
            inds = list(inds[0]) # [stim1_inds, stim2_inds]
            dists_full_intrvl = list(scalar_dists(tmp_neuron, inds, time_before_stim, time_after_stim))

            dists_full_intrvl = [monkey, session, task, condition, lobe, roi, j, row[0], row[1], row[2], row[3]] + [dists_full_intrvl]
            labels_full_intrvl = ['monkey', 'session', 'task', 'condition', 'lobe', 'roi', 'neuron_num',
            'x1', 'y1', 'x2', 'y2', 'isi_full', 'spike_full', 'sync_full']
            t0 = time_before_stim
            t1 = t0 + bin_size #initialize t0 & t1 on first step
            while t1 <= time_after_stim: 
                dists_full_intrvl.append(scalar_dists(tmp_neuron, inds, t0, t1))
                labels_full_intrvl.append(['isi_full_' + str(t1)[0:4], 'spike_full_' + str(t1)[0:4], 'sync_full_' + str(t1)[0:4]])
                t0 += step_size
                t1 = t0 + bin_size
            dists_full_intrvl = np.hstack(dists_full_intrvl)
            labels_full_intrvl1 = np.hstack(labels_full_intrvl)
            three_dists.append(dists_full_intrvl)
    three_dists = np.vstack(three_dists)
    three_dists_pd = pd.DataFrame(three_dists, columns=labels_full_intrvl1)
    return three_dists_pd

def main(file_name_i):
    mat = scipy.io.loadmat(file_name_i, squeeze_me=True, struct_as_record=True)

    try:
        sess_num = re.search(r'(/)([0-9]*)(.mat)',file_name_i).group(2)
    except:
        sess_num = re.search(r'(/)([0-9]*_[0-9]*)(.mat)',file_name_i).group(2)

    unitInfo = pd.read_csv(parent_save_dir + '/unitInfo' + '/' + sess_num + '.txt', sep=',')

    trialInfo_allTasks = pd.read_csv(parent_save_dir + '/trialInfo' + '/' + sess_num + '.txt', sep=',')

    trialInfo = pd.read_csv(parent_save_dir + '/mini_data/trial_info' + sess_num + '.csv', sep=',')

    if trialInfo.colorCoherence[0] != 1:  # they played around with color & motion coherence in a few sessions (110106 to 110115_01) (i.e., discrimination task)
        print(sess_num)
        return

    sessInfo = pd.read_csv(parent_save_dir + '/mini_data/sess_info' + sess_num + '.csv', sep=',')
    monkey = sessInfo.iloc[0,0]
    badTrials = trialInfo_allTasks.badTrials.copy()
    spikes = mat['spikeTimes'].copy()
    spikes = spikes[trialInfo_allTasks.task=='mocol',:]
    badTrials = badTrials[trialInfo_allTasks.task=='mocol']
    spikes = spikes[badTrials==0,:]
    condition = trialInfo.rule.copy()
    spikes_motion = spikes[condition=='motion']
    labels_motion = trialInfo[['color','direction']].copy() #color & direction coordinates
    labels_motion = labels_motion[condition=='motion']
    spikes_color = spikes[condition=='color']
    labels_color = trialInfo[['color','direction']].copy() #color & direction coordinates
    labels_color = labels_color[condition=='color']
    isolated_neurons = unitInfo.isolated
    other_args = (unitInfo, monkey)


    try:
        if os.path.isfile(save_dir + '/motion_' + sess_num + '.csv'):
            print(save_dir + '/motion_' + sess_num + '.csv' + ' already exists')
            exit()
        dists_motion = compute_dists(spikes_motion, labels_motion, isolated_neurons, ref_dists, other_args, 'motion')
        dists_motion.to_csv(save_dir + '/motion_' + sess_num + '.csv')
    except:
        print(sess_num, 'motion error')
    try:
        if os.path.isfile(save_dir + '/color_' + sess_num + '.csv'):
            print(save_dir + '/color_' + sess_num + '.csv' + ' already exists')
            exit()
        dists_color = compute_dists(spikes_color, labels_color, isolated_neurons, ref_dists, other_args, 'color')
        dists_color.to_csv(save_dir + '/color_' + sess_num + '.csv')
    except:
        print(sess_num, 'color error')

parent_save_dir = '/home/user/../data'
save_dir = parent_save_dir + '/save_dir'
data_dir = '/home/user/data'
mat_files_match = data_dir + '/*.mat'
data_files = glob.glob(mat_files_match)
data_files.sort()
ref_dists = pd.read_csv(parent_save_dir + '/reference_dists.csv', sep=',')


focus_on_time = 0
bin_size = 0.05
time_after_stim = 0.2 + bin_size 
time_before_stim = 0.0 
step_size = 0.01
edges = [-1.5,3.5]
on_fixation_cross = -1.5


njobs = 27
Parallel(n_jobs=njobs)( delayed(main) (file_name_i) for file_name_i in data_files )

# get stimuli coordinates matrix
coords = [(1,1),(2,1),(3,1),(4,1),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3),(1,4),(2,4),(3,4),(4,4)]
matrix_ne = torch.zeros((16,16))
matrix_sim = torch.zeros((16,16))
for i in range(len(coords)):
    for j in range(len(coords)):
        matrix_ne[i,j] = np.linalg.norm(np.array(coords[i])-np.array(coords[j]))
torch.save(matrix_ne, parent_save_dir+ f'stimuli_coordinates_matrix.pt')


