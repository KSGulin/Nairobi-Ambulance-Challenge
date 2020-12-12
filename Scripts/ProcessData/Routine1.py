import numpy as np
import pandas as pd
import importlib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

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
cl_train_inds = np.argwhere(total_binary_crashes > 4)
cl_train_labels = u_crash_labels_binary[cl_train_inds].squeeze()
cl_train_data = u_predictive_data[cl_train_inds].squeeze()

#Logistic
logistic = RM.Logistic(C = 1)
ridge = RM.RoundedRidge(alpha = 1)
params = [10**p for p in range(-7, 7)]

#RandomForest
# n_estimators = [10**p for p in range(1,4)]
# max_depth = [1 ,2, 3, 4, 5, 6, None]
# params = [n_estimators, max_depth]

# #SVM
# C = [10**p for p in range(-5,4)]
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# params = [C, kernel]

# #NNC
# layers = [2,3]
# activations = ['relu']
# epochs = [2, 4, 6, 8, 10]
# batch_size = [2, 5, 10, 20]
# weights = [{0:1, 1:100}]
# params = [layers, activations, epochs, batch_size, weights]


# plt.style.use('ggplot')
# sns.set_style("ticks")


# sns.pairplot(df_upsampled, hue="Crash")
# plt.show()

#params = Opt.Param_Permutations(params)

# combined_data = np.insert(cl_train_data[1], 0, cl_train_labels[1], axis=1)
# df = pd.DataFrame(data=combined_data, columns=['Crash', 'Time of Day', 'Precipitation', 'Relative Humidity', 'Specific Humidity', 'Temperature', 'u Wind', 'v Wind'])
# df_nc = df[df['Crash']==0]
# df_c = df[df['Crash']==1]
# df_c_upsampled = resample(df_c, replace=True, n_samples=len(df_nc))   
# df_upsampled = pd.concat([df_nc, df_c_upsampled])

# upsampled_data = df_upsampled.to_numpy()
# labels = upsampled_data[:,0]
# data = upsampled_data[:,1:]

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators = 1000, max_depth = 10, oob_score = True)
# model.fit(data, labels)

# combined_data = np.insert(cl_train_data[4], 0, cl_train_labels[4], axis=1)
# df = pd.DataFrame(data=combined_data, columns=['Crash', 'Time of Day', 'Precipitation', 'Relative Humidity', 'Specific Humidity', 'Temperature', 'u Wind', 'v Wind'])
# df_nc = df[df['Crash']==0]
# df_c = df[df['Crash']==1]
# df_c_upsampled = resample(df_c, replace=True, n_samples=len(df_nc))   
# df_upsampled = pd.concat([df_nc, df_c_upsampled])

# upsampled_data = df_upsampled.to_numpy()
# labels = upsampled_data[:,0]
# data = upsampled_data[:,1:]

# model = RandomForestClassifier(n_estimators = 1000, max_depth = None, oob_score = True)
# model.fit(data, labels)



for i in range(len(cl_train_labels)):

    performance, model = Opt.Cross_Validation_Tune(CM.Logistic, cl_train_data[i], cl_train_labels[i], params)


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