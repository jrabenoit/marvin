#!/usr/bin/env python3

import pprint, itertools, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

def OuterCv():        

    with open('pickles/data_dict.pickle', 'rb') as f:
        data_dict= pickle.load(f)

    X= np.array(data_dict['subject_id'])
    y= np.array(data_dict['labels'])
    
    X_train, X_test, y_train, y_test= [], [], [], []      

    outer = StratifiedKFold(y, n_folds=5)
    for train_index, test_index in outer:
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    
    outer_cv= {'X_train': np.array(X_train),
               'X_test': np.array(X_test),
               'y_train': np.array(y_train),
               'y_test': np.array(y_test)}

    with open('pickles/outer_cv.pickle', 'wb') as f:
        pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    '''Set up as a flat structure of 25 df'''
    with open('pickles/outer_cv.pickle', 'rb') as f:
        outer_cv= pickle.load(f)
    
    X= outer_cv['X_train']
    y= outer_cv['y_train']
    
    X_train, X_test, y_train, y_test = [], [], [], []

    #read loop as, "for each pair of X and y lists in (X,y)"
    for X_, y_ in zip(X, y): 
        inner = StratifiedKFold(y_, n_folds=5)
        for train_index, test_index in inner:      
            X_train.append(X_[train_index])
            X_test.append(X_[test_index])
            y_train.append(y_[train_index])
            y_test.append(y_[test_index]) 

    inner_cv= {'X_train': X_train,
               'X_test': X_test,
               'y_train': y_train,
               'y_test': y_test}

    with open('pickles/inner_cv.pickle', 'wb') as f:
        pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return

    
