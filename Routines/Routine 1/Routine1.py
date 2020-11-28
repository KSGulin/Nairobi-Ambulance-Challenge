import numpy as np
import pandas as pd
import importlib
import sys

sys.path.insert(1, '.')

import Support as Sup
import Clustering as Cl
import Optimization as Opt
import PreProcessing as Proc
import FeatureSelection as FS
import RegressionModels as RM
import ClassificationModels as CM

## Load data
crashes = Sup.Load_Crash_Data()
crash_locations = np.delete(crashes, 0, 1).astype(float)
crash_locations[:, [1, 0]] = crash_locations[:, [0, 1]]
weather = pd.read_csv('Data/Weather_Nairobi_Daily_GFS_interpolated.csv', parse_dates=['Date'])
weather_train = weather.iloc[:546,]
weather_test = weather.iloc[546:,]

predictive_data = np.load('Routines/Routine 1/predictive_data.npy')
crash_labels = np.load('Routines/Routine 1/crash_labels.npy')
inds_unmerged = np.load('Routines/Routine 1/inds_unmerged.npy')
segments_merged = pd.read_pickle('Routines/Routine 1/segments_merged.pkl')
u_predictive_data = np.load('Routines/Routine 1/u_predictive_data.npy')
u_crash_labels = np.load('Routines/Routine 1/u_crash_labels.npy')
u_crash_labels_binary = (u_crash_labels > 0)

total_binary_crashes = np.sum(u_crash_labels_binary, axis = 1)
cl_train_inds = u_crash_labels_binary[total_binary_crashes > 4]
cl_train_labels = u_crash_labels_binary[cl_train_inds]
cl_train_data = u_predictive_data[cl_train_inds]

logistic = RM.Logistic(C = 1)
ridge = RM.RoundedRidge(alpha = 1)
params = [10**p for p in range(-7, 7)]
n_estimators = [10**p for p in range(1,3)]
max_depth = [1 ,2, 3, 4, 5, 6]
params = [n_estimators, max_depth]

for i in range(len(cl_train_labels)):
    performance, model = Opt.Cross_Validation_Tune(CM.RandomForestCl, cl_train_data[i], cl_train_labels[i], params)


## Generate data structs from the start
# segments_merged = Sup.Load_Merge_Road()
# segments_merged, inds_unmerged = Proc.Merge_Roads_Crashes(segments_merged, crash_locations, crashes)
# predictive_data, crash_labels = Proc.Create_Time_Series_Data(segments_merged, weather_train)
# grid = Proc.Create_Grid(crashes[inds_unmerged,:], .1)
# flat_grid = [j for sub in grid for j in sub]
# u_predictive_data, u_crash_labels, active_cells = Proc.Create_Unmerged_Series(flat_grid, weather_train)

## Feature Selection
trimmed = FS.Nan_Threshold(predictive_data, .6)
trimmed_20 = FS.Top_Variance_Select(trimmed, 20)