#!/usr/bin/env python 

import pprint, itertools
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

def oSkfCv(data, labels):        
    '''Outer loop 5-fold CV. X = data, y = label'''
    X = [v for (k,v) in sorted(data.items())]
    y = [v for (k,v) in sorted(labels.items())]
    
    skf = StratifiedKFold(labels, n_folds=5)
        for train, test in skf:
            oX_train[i].append(data[i])
            oX_test[i].append(data[i])
            oy_train[i].append(labels[i])
            oy_test[i].append(labels[i])          

    return oX_train, oX_test, oy_train, oy_test
    
#Do 5-fold CV in inner loop
def iSkfCv(oy_train, oX_train, iter_n):
    '''Set up as a flat structure of 25 lists'''
    iX_train = defaultdict(list)
    iX_test = defaultdict(list)
    iy_train = defaultdict(list)
    iy_test= defaultdict(list)

    train_index_inner = defaultdict(list)
    test_index_inner = defaultdict(list)
#Some subjects will not be listed here because they are in the holdout set.
    for i in range(iter_n):
        for oX_train_, oy_train_ in zip(oX_train[i], oy_train[i]):
            iskf = StratifiedKFold(oy_train_, n_folds=5)
            for train_index, test_index in iskf:      
                train_index_inner[i].append(train_index)
                test_index_inner[i].append(test_index)
                iX_train[i].append(oX_train_[train_index])
                iX_test[i].append(oX_train_[test_index])
                iy_train[i].append(oy_train_[train_index])
                iy_test[i].append(oy_train_[test_index])
    return iX_train, iX_test, iy_train, iy_test, train_index_inner, test_index_inner

    
