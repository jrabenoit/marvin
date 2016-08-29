#!/usr/bin/env python 

import copy
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model

# .fit fits the model to the dataset in brackets. 
# Score tests the fitted model on data.

def GauNaiBay(df_inner_cv):
    X_train = df_inner_cv['X_train_trans']
    X_test = df_inner_cv['X_test_trans']
    y_train = df_inner_cv['y_train_trans']
    y_test = df_inner_cv['y_test_trans']
    
    for i in range(len(X_train)):
        gnb = naive_bayes.GaussianNB()
        gnb.fit(X_train[i], y_train[i])
        df_inner_cv['X_train_est_score'][i] = gnb.score(X_train[i], y_train[i])
        df_inner_cv['X_test_est_score'][i] = gnb.score(X_test[i], y_test[i]) 
        df_inner_cv['X_train_est_predict']= gnb.predict(X_train[i])
        df_inner_cv['X_test_predict']= gnb.predict(X_test[i])
    
    return df_inner_cv

'''    
def KNeighbors(X_train, X_test, y_train, y_test):
    for i in range(0,len(X_train)):
        knc = neighbors.KNeighborsClassifier()
        knc.fit(X_train[i], y_train[i])
        X_train[i] = knc.score(X_train[i], y_train[i])
        X_test[i] = knc.score(X_test[i], y_test[i])
    return X_train, X_test

def CSupSvc(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        csvm = svm.SVC()
        csvm.fit(X_train[i], y_train[i])
        X_train[i] = csvm.score(X_train[i], y_train[i])
        X_test[i] = csvm.score(X_test[i], y_test[i])
    return X_train, X_test
    
def RandomForest(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        rf = ensemble.RandomForestClassifier()
        rf.fit(X_train[i], y_train[i])
        X_train[i] = rf.score(X_train[i], y_train[i])
        X_test[i] = rf.score(X_test[i], y_test[i])
    return X_train, X_test

def ExtraTrees(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        rf = ensemble.ExtraTreesClassifier()
        rf.fit(X_train[i], y_train[i])
        X_train[i] = rf.score(X_train[i], y_train[i])
        X_test[i] = rf.score(X_test[i], y_test[i])
    return X_train, X_test

def LinearSgd(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        sgd = linear_model.SGDClassifier()
        sgd.fit(X_train[i], y_train[i])
        X_train[i] = sgd.score(X_train[i], y_train[i])
        X_test[i] = sgd.score(X_test[i], y_test[i])
    return X_train, X_test
'''    
