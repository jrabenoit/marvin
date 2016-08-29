#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath, nitime, re
from nitime.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
from scipy import stats

def Setup(file_directory): 

    files= glob.glob(file_directory + '/*.mat')
    subjects, labels, voxels, tr, data= [], [], [], [], []
    site_tr = {1:2, 2:2.3, 3:2, 4:2.52, 5:2.4,\
               6:1.25, 7:2, 8:2.4, 9:2.5, 10:2, 11:2}

    for i in files:
        subjects.append(ntpath.basename(os.path.splitext(i)[0]))
        if 'HC' in i: labels.append(0)
        elif 'MDD' in i: labels.append(1)
        mat_table= sio.loadmat(i)
        float_vox= np.asarray(mat_table['PAC_data'], dtype= np.float64)
        voxels.append(float_vox)
        site=(list(map(int, re.findall('\d+',i)))[1])
        subj_tr = [v for k,v in site_tr.items() if k==site][0]
        tr.append(subj_tr)
        nitime_vox= TimeSeries(float_vox, sampling_interval= subj_tr)
        norm_smoothed_vox = ...#START HERE
        zscored_vox= stats.zscore(nitime_vox)
        data.append(zscored_vox)
        
    
    df = {'subjects': subjects, 
          'labels': labels, 
          'voxels':voxels,
          'tr':tr,
          'data': data}
    
    df_subjects= pd.DataFrame(df, index= list(range(len(files))))

    return df_subjects
