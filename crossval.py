#!/usr/bin/env python 

import pprint, itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

def OuterCv(subjects):        
    X= subjects['data']
    y= subjects['labels']

    X_train_outer, X_test_outer, y_train_outer, y_test_outer = [], [], [], []      

    outer = StratifiedKFold(y, n_folds=5)
    for train_index, test_index in outer:
        X_train_outer.append(X[train_index])
        X_test_outer.append(X[test_index])
        y_train_outer.append(y[train_index])
        y_test_outer.append(y[test_index])

    content= {'X_train': X_train_outer,
              'X_test': X_test_outer,
              'y_train': y_train_outer,
              'y_test': y_test_outer}
    
    df_outer_cv= pd.DataFrame(content, index= list(range(5)))

    return df_outer_cv
    
def InnerCv(df_outer_cv):
    '''Set up as a flat structure of 25 lists'''
    X= df_outer_cv['X_train']
    y= df_outer_cv['y_train']
    
    X_train_inner, X_test_inner, y_train_inner, y_test_inner = [], [], [], []

    for X_, y_ in zip(X, y): #read as, "for each pair of X and y lists in (X,y)"
        inner = StratifiedKFold(y_, n_folds=5)
        for train_index, test_index in inner:      
            X_train_inner.append(X_[train_index])
            X_test_inner.append(X_[test_index])
            y_train_inner.append(y_[train_index])
            y_test_inner.append(y_[test_index]) 

    content= {'X_train': X_train_inner,
              'X_test': X_test_inner,
              'y_train': y_train_inner,
              'y_test': y_test_inner}

    df_inner_cv= pd.DataFrame(content, index= list(range(25)))
 
    return df_inner_cv

    
