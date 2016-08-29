#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath
from sklearn.preprocessing import StandardScaler
from scipy import stats

def Setup(file_directory): 

    files= glob.glob(file_directory + '/*.mat')
    subjects, labels, voxels, data= [], [], [], []

    for i in files:
        subj_code= os.path.splitext(i)[0]
        subjects.append(ntpath.basename(subj_code))
        if 'HC' in i: labels.append(0)
        elif 'MDD' in i: labels.append(1)
        mat_table= sio.loadmat(i)
        raw_vox= mat_table['PAC_data']
        float_vox= np.asarray(raw_vox, dtype= np.float64)
        voxels.append(float_vox)
        zscored_vox = stats.zscore(float_vox)
        data.append(zscored_vox)
    
    df = {'subjects': subjects, 
          'labels': labels, 
          'voxels':voxels,
          'data': data}
    
    df_subjects= pd.DataFrame(df, index= list(range(len(files))))

    return df_subjects
