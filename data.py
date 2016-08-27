#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath
from sklearn import preprocessing

def Setup(file_directory): 
    filenames= file_directory + '*.mat'
    sortedfiles= glob.glob(filenames)
    hcfiles= [s for s in sortedfiles if 'HC' in s]
    hcfiles.sort()
    mddfiles= [s for s in sortedfiles if 'MDD' in s]
    mddfiles.sort()
    files= hcfiles + mddfiles

    indices = list(range(0,len(files)))
    subjects= []
    for i in files:
        j= os.path.splitext(i)[0]
        subjects.append(ntpath.basename(j))
    labels= []
    for i in files:
        if 'HC' in i: labels.append(0)
        elif 'MDD' in i: labels.append(1)
    vectors= []
    for i in files:
        mat= sio.loadmat(i)
        matvector= np.reshape(mat['PAC_data'],(1,np.product(\
                   mat['PAC_data'].shape)))
        vectors.append(matvector)
    data= []
    zscaler= preprocessing.StandardScaler()
    for i in vectors:
        zdata = zscaler.fit_transform(vectors[i])
        data.append(zdata)

    df= {}
    df['subjects']= pd.Series(subjects, index= indices)
    df['labels']= pd.Series(labels, index= indices)
    df['vectors']= pd.Series(vectors, index= indices)
    df['data']= pd.Series(data, index= indices)

    return df
