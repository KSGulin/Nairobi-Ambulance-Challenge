import numpy as np
import pandas as pd
import importlib
import sys
from scipy import stats
sys.path.insert(1, '.')

import Support as Sup
import PreProcessing as Proc
import Optimization as Opt
import FeatureSelection as FS
import RegressionModels as RM
import ClassificationModels as CM

full_paired_data = np.load('Scripts/ProcessData/full_paired_data.npy')
paired_crash_labels = np.load('Scripts/ProcessData/paired_crash_labels.npy')
paired_binary_labels = (paired_crash_labels > 0)

full_paired_data = FS.Nan_Threshold(full_paired_data, .6)

full_paired_data, nan_dict = Proc.Replace_Nan_Mode(full_paired_data)

featlen = full_paired_data.shape[1]
cramers = np.zeros(featlen)
cramers_nan = np.zeros(len(nan_dict))
pearsons = np.zeros(6)

for i in range(6):
    pearsons[i] = stats.pearsonr(full_paired_data[:,i], paired_binary_labels.astype(int))[0]

# for i in range(6,featlen):
#     cramers[i] = FS.Calculate_Cramers_Corrected(full_paired_data[:,i], paired_binary_labels.astype(int))

# for i,key in enumerate(nan_dict.keys()):
#     cramers_nan[i] = FS.Calculate_Cramers_Corrected(nan_dict[key], paired_binary_labels.astype(int))

# threshold = np.percentile(cramers, 90)
# ind = np.argwhere(cramers > threshold)
# ind = np.concatenate(([0, 1, 2, 3, 4, 5], ind.squeeze()))
# trimmed_data = full_paired_data[:, ind]

# # for i in ind:
# #     if i in nan_dict:
# #         trimmed_data = np.insert(trimmed_data, trimmed_data.shape[1], nan_dict[i], axis = 1)


# trimmed_data = np.load('Scripts/PairedData/paired_trimmed_data.npy')



# # Random forest
# n_estimators = [10**p for p in range(1,4)]
# max_depth = [5, 10, 20, 50, 100, None]
# params = [n_estimators, max_depth]
# params = Opt.Param_Permutations(params)

# models, performance = Opt.Cross_Validate_Multiple(CM.RandomForestCl, [trimmed_data], [paired_binary_labels], params, upsample = 1)

simple_paired_data = np.load('Scripts/ProcessData/simple_paired_data.npy')
simple_crash_labels = np.load('Scripts/ProcessData/simple_crash_labels.npy')


simple_paired_data = FS.Nan_Threshold(simple_paired_data, .6)
simple_paired_data, nan_dict = Proc.Replace_Nan_Mode(simple_paired_data)

for key in nan_dict.keys():
    simple_paired_data = np.insert(simple_paired_data, simple_paired_data.shape[1], nan_dict[key], axis = 1)

# Lasso
alpha = [10**p for p in range(-8,4)]
standardize = [True, False]
params = [alpha, standardize]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Regression(RM.RoundedLasso, [simple_paired_data], [simple_crash_labels], params, upsample = 0)

## Feature Reduction
ind_delete = np.where(models[0].model.coef_ == 0)[0]
simple_trimmed_data = np.delete(simple_paired_data, ind_delete, axis = 1)

simple_trimmed_data = np.load('Scripts/PairedData/simple_trimmed_data.npy')
simple_crash_labels = np.load('Scripts/ProcessData/simple_crash_labels.npy')
binary_labels = (simple_crash_labels > 0)


# Logistic regression
# params = [10**p for p in range(-7, 7)]

# models, performance = Opt.Cross_Validate_Multiple(CM.Logistic, [simple_trimmed_data], [binary_labels], params)

# Random forest
# n_estimators = [10**p for p in range(1,4)]
# max_depth = [2, 5, 10, 20, 50, 100]
# params = [n_estimators, max_depth]
# params = Opt.Param_Permutations(params)

# models, performance = Opt.Cross_Validate_Multiple(CM.RandomForestCl, [simple_trimmed_data], [binary_labels], params)

# #SVM
# C = [10**p for p in range(-5,4)]
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# params = [C, kernel]
# params = Opt.Param_Permutations(params)

# models, performance = Opt.Cross_Validate_Multiple(CM.SVM, [simple_trimmed_data], [binary_labels], params)

# Random forest regression
n_estimators = [10**p for p in range(1,4)]
max_depth = [2, 5, 10, 20, 50, 100]
params = [n_estimators, max_depth]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Regression(RM.RandomForrestReg, [simple_trimmed_data], [simple_crash_labels], params, upsample = 0)

print('done')


