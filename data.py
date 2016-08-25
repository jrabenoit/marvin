#!/usr/bin/env python 

import os, csv, glob, pprint, random, copy
import scipy.io as sio
import numpy as np
import pandas as pd

#Goals:
# - Generate an ordered list of filenames (DONE)
# - Create an ordered list of labels (0 or 1 (if file contains HC, let label =0 else =1) (DONE)
# - Go into the file and extract the 'PAC_data' array for each subject (DONE)
# - Concatenate this extracted array into a 2D matrix with each column repping a feature & each row a subject (DONE, 228436x135 ARRAY RESHAPED TO VECTOR)
# - Have these two lists feed into current stack of cross validation (IN PROGRESS)

def Groups():
    '''Insert location of data directory in DATA_DIRECTORY'''    
    
    DATA_DIRECTORY = '/home/james/Desktop/PAC Data/pac_2016_data_files/' 
    
    filedir = DATA_DIRECTORY + '*.mat'
    filenames = glob.glob(filedir)
    hcfiles = [s for s in filenames if 'HC' in s]
    hcfiles.sort()
    mddfiles = [s for s in filenames if 'MDD' in s]
    mddfiles.sort()
    datafiles = hcfiles + mddfiles

    dataarrays = {}
    for f in datafiles:
        df = sio.loadmat(f)
        df2 = np.reshape(df['PAC_data'],(1,np.product(df['PAC_data'].shape)))
        dataarrays[f] = df2
    
    grouplabels = {}
    for f in datafiles:
        if 'HC' in f: grouplabels[f] = 0
        if 'MDD' in f: grouplabels[f]= 1

#grouplabels[f] = np.append(np.zeros(len(hcfiles),dtype=np.int8),np.ones(len(mddfiles),dtype=np.int8))


    return dataarrays, grouplabels
    
    


    
