#!/usr/bin/env python 

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, pprint, random, copy
from sklearn import preprocessing

def Setup(datadirectory): 
    filenames = datadirectory + '*.mat'
    sortedfiles = glob.glob(filenames)
    hcfiles = [s for s in sortedfiles if 'HC' in s]
    hcfiles.sort()
    mddfiles = [s for s in sortedfiles if 'MDD' in s]
    mddfiles.sort()
    
    files = hcfiles + mddfiles

    vectors = {}
    for i in files:
        mat = sio.loadmat(i)
        matvector = np.reshape(mat['PAC_data'],(1,np.product(mat['PAC_data'].shape)))
        vectors[i] = matvector

    data = {}
    zscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    for i in vectors:
        data[i] = zscaler.fit_transform(vectors[i])

    labels = {}
    for i in datafiles:
        if 'HC' in i: labels[i] = 0
        elif 'MDD' in i: labels[i]= 1

    return data, labels
