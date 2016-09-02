#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath, re, scipy.signal
from sklearn.preprocessing import StandardScaler
from scipy import stats

def Setup(file_directory): 

    files= glob.glob(file_directory + '/*.mat')
    subject_id= np.array(range(len(files)))
    site_tr = {1:2, 2:2.3, 3:2, 4:2.52, 5:2.4,\
        6:1.25, 7:2, 8:2.4, 9:2.5, 10:2, 11:2}
    subject, labels, voxels, tr= [], [], [], []
    n_scans, scantime, samples, resampled= [], [], [], []
    aligned, data= [], []

    for i in files:
        subject.append(ntpath.basename(os.path.splitext(i)[0]))
        if 'HC' in i: labels.append(0)
        elif 'MDD' in i: labels.append(1)
        float_vox= np.asarray(sio.loadmat(i)['PAC_data'], dtype= np.float64)
        voxels.append(float_vox)
        site=(list(map(int, re.findall('\d+',i)))[1])
        scan_tr= [v for k,v in site_tr.items() if k==site][0]
        tr.append(scan_tr)
        volumes= float_vox.shape[1]
        n_scans.append(volumes)
        scan_duration= volumes*scan_tr
        scantime.append(scan_duration)
        n_samples= int(scan_duration/max(site_tr.values()))        
        samples.append(n_samples)
        resampled_x= scipy.signal.resample(float_vox, n_samples, axis=1)
        resampled.append(resampled_x)

    for i in range(len(files)):
        min_sample_len= min(samples)
        cutoff_vox= resampled[i][:,:min(samples)]
        aligned.append(cutoff_vox)
        vector_data= np.reshape(cutoff_vox,(1,-1))[0]
        data.append(vector_data)        
    
    data_dict = {       'id': subject_id,
                   'subject': subject, 
                    'labels': labels, 
                    'voxels': voxels,
                        'tr': tr,
                   'n_scans': n_scans,
                  'scantime': scantime,
                   'samples': samples,
                 'resampled': resampled,
                   'aligned': aligned,
                      'data': data}

    return data_dict
