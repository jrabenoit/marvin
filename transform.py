#!/usr/bin/env python 

from sklearn import decomposition
import copy
    
def NullDecomp(df_inner_cv):
    df_inner_cv['X_train_trans']= df_inner_cv['X_train_feat']
    df_inner_cv['X_test_trans']= df_inner_cv['X_test_feat']
    df_inner_cv['y_train_trans']= df_inner_cv['y_train_feat']
    df_inner_cv['y_test_trans']= df_inner_cv['y_test_feat']

    return df_inner_cv

'''    
def RPca(iX_train, iX_test, iy_train, iy_test, n_components=3):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    for i in range(0,len(iX_train)):
        pca = decomposition.RandomizedPCA(n_components=n_components)
        pca.fit(dX_train[i])
        dX_train[i] = pca.transform(dX_train[i])
        dX_test[i] = pca.transform(dX_test[i])
    return dX_train, dX_test, dy_train, dy_test
'''
