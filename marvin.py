#!/usr/bin/env python 

import pprint, itertools
import data,crossval,iterator,comparator,bootstrap,visualize
from collections import defaultdict

file_directory = '/home/james/Desktop/PAC Data/pac_2016_data_files/'

#Select a group of scans to use 
print('Step 1: Create dataframe with subject & group identifiers)
data, labels= data.Setup(file_directory)

print('Step 2: Set Up Outer CV')
oX_train, oX_test, oy_train, oy_test= crossval.oSkfCv(df)

print('Step 4: Set Up Inner CV from Outer CV training set')
iX_train, iX_test, iy_train, iy_test= crossval.iSkfCv(oX_train, oy_train)

print('Step 7: Try all featsel/decomp/mltool Combos')
test_results, param_set_list = iterator.ParameterSets(iX_train, iX_test, iy_train, iy_test, iter_n)

print('Step 8: Pick Best featsel/decomp/mltool Combo')
fold_index, folds = comparator.PickBest(test_results)

print('Step 9: Run Best Combo on Outer CV Holdout')
final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index, iter_n) 

print('Step 10: Print Test vs. Chance Results')
final_train_correct, final_test_correct = comparator.PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n, train_index_files, test_index_files)

#Build accuracy profile for each subject
subject_results_test, subject_results_train = comparator.SubjectAccuracy(iter_n, final_train_correct, final_test_correct, train_index_files, test_index_files, concat_subjects_dict)

#Concatenate classification attempts for each subject
for key, value in subject_results_test.items():
    concatenated_test[key].append(value)

for key, value in subject_results_train.items():
    concatenated_train[key].append(value)

print('>>>CHAINING TRAINING RESULTS TOGETHER')
concatenated_train_chained = defaultdict(list)
for key, value in concatenated_train.items():
    concatenated_train_chained[key] = list(itertools.chain.from_iterable(value))
    
#print('>>>CHAINING TEST RESULTS TOGETHER')
concatenated_test_chained = defaultdict(list)
for key, value in concatenated_test.items():
    concatenated_test_chained[key] = list(itertools.chain.from_iterable(value))

print('>>>TRAIN SUBJECT ACCURACY SCORES')
per_subject_train_acc = defaultdict(list)
for key, value in concatenated_train_chained.items():
    per_subject_train_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_train_acc)

print('>>>TEST SUBJECT ACCURACY SCORES')
per_subject_test_acc = defaultdict(list)
for key, value in concatenated_test_chained.items():
    per_subject_test_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_test_acc)

final_acc = sum(list(per_subject_test_acc.values()))/len(list(per_subject_test_acc.values()))
print('\n>>>AVERAGE ACCURACY: {}%'.format(round(final_acc,2)))
p_value = bootstrap.EmpiricalDistro(n_1, n_2, per_subject_test_acc)
print('>>>P VALUE UNCORRECTED: {}'.format(p_value))
print('>>>P VALUE CORRECTED: {}'.format(p_value*3))

print('>>>SAMPLE SIZE: {}'.format(n_1 + n_2))

