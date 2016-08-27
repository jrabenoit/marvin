#!/usr/bin/env python 

import pprint, itertools
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

def oSkfCv(df):        
    '''Outer loop 5-fold CV. X = data, y = label'''
    X = df['data']
    y = df['labels']
    oX_train, oX_test, oy_train, oy_test= [], [], [], []    

    skf = StratifiedKFold(y, n_folds=5)
    for train_index, test_index in skf:
        oX_train.append(X[train_index])
        oX_test.append(X[test_index])
        oy_train.append(y[train_index])
        oy_test.append(y[test_index])
    print('TRAIN:', train_index, 'TEST:', test_index)    
 
    return oX_train, oX_test, oy_train, oy_test
    
#Do 5-fold CV in inner loop
def iSkfCv(oX_train, oy_train):
    '''Set up as a flat structure of 25 lists'''
    iX_train, iX_test, iy_train, iy_test= [], [], [], []

    for oX_train_, oy_train_ in zip(oX_train[i], oy_train[i]):
        iskf = StratifiedKFold(oy_train_, n_folds=5)
        for train_index, test_index in iskf:      
            iX_train[i].append(oX_train_[train_index])
            iX_test[i].append(oX_train_[test_index])
            iy_train[i].append(oy_train_[train_index])
            iy_test[i].append(oy_train_[test_index])
    return iX_train, iX_test, iy_train, iy_test, train_index_inner, test_index_inner

    
