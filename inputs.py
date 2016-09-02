#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath, re, scipy.signal
from sklearn.preprocessing import StandardScaler
from scipy import stats

def DictSetup(file_directory): 
    '''Creates dict for scan data, generates file list & subject ID'''
    subjects, labels, voxels, scan_tr, n_scans= [], [], [], [], []
    scantime, samples, resampled, aligned, data= [], [], [], [], []
    files= glob.glob(file_directory + '/*.mat')
    subject_id= np.array(range(len(files)))
    data_dict = {     'files': files,
                 'subject_id': subject_id,
                   'subjects': subjects, 
                     'labels': labels, 
                     'voxels': voxels,
                    'scan_tr': scan_tr,
                    'n_scans': n_scans,
                   'scantime': scantime,
                    'samples': samples,
                  'resampled': resampled,
                    'aligned': aligned,
                       'data': data}
   
    return data_dict

def LabelSetup(data_dict):
    '''Labels data by group, finds scan location & TR'''
    site_tr = {1:2, 2:2.3, 3:2, 4:2.52, 5:2.4, 6:1.25,\
               7:2, 8:2.4, 9:2.5, 10:2, 11:2}
    for i in data_dict['files']:
        data_dict['subjects'].append(ntpath.basename(os.path.splitext(i)[0]))
        if 'HC' in i: data_dict['labels'].append(0)
        elif 'MDD' in i: data_dict['labels'].append(1)
        site= (list(map(int, re.findall('\d+',i)))[1])
        scan_tr= [v for k,v in site_tr.items() if k==site][0]
        data_dict['scan_tr'].append(scan_tr)       

    return data_dict
    
def ScanLoader(data_dict):
    '''Loads scan voxels, gets # of scans, scan duration'''
    for i in range(len(data_dict['files'])):
        float_vox= np.asarray(sio.loadmat(data_dict['files'][i])['PAC_data'],\
                              dtype= np.float64)
        data_dict['voxels'].append(float_vox)
        n_volumes= float_vox.shape[1]
        data_dict['n_scans'].append(n_volumes)
        scan_duration= n_volumes*data_dict['scan_tr'][i]
        data_dict['scantime'].append(scan_duration)        

    return data_dict
    
def Resample(data_dict):
    '''Gets # of samples to take per scan, resamples based on max TR'''
    trmax= max(data_dict['scan_tr'])
    for i in range(len(data_dict['files'])):
        n_samples= int(data_dict['scantime'][i]/trmax)        
        data_dict['samples'].append(n_samples)
        resampled_x= scipy.signal.resample(data_dict['voxels'][i],\
                                           n_samples, axis=1)
        data_dict['resampled'].append(resampled_x)

    return data_dict

def Align(data_dict):
    '''Cuts scans to same sample length at max TR sampling rate'''
    min_sample_len= min(data_dict['samples'])
    for i in range(len(data_dict['files'])):
        cutoff_vox= data_dict['resampled'][i][:,:min_sample_len]
        data_dict['aligned'].append(cutoff_vox)

    return data_dict
    
def Flatten(data_dict):
    '''Vectorizes data into a (1, -1) shape'''
    for i in range(len(data_dict['files'])):
        vector_data= np.reshape(data_dict['aligned'][i],(1,-1))[0]
        data_dict['data'].append(vector_data)        
    
    return data_dict
    
