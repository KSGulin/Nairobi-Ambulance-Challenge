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

## Load Data
pre_unpaired_data = np.load('Scripts/ProcessData/pre_unpaired_data.npy')
pre_unpaired_labels = np.load('Scripts/ProcessData/pre_unpaired_labels.npy')

## Convert to binary labels for each 3-hour period
binary_labels = (pre_unpaired_labels > 0)
binary_label_sums = np.sum(binary_labels, axis = 1)

## Create a dataframe with the data and labels (for visualization purposes)
combined_dfs = []
for i in range(len(binary_labels)):
    combined_data = np.insert(pre_unpaired_data[i], 0, binary_labels[i], axis=1)
    combined_dfs.append(pd.DataFrame(data=combined_data, 
        columns=['Crash', 'Time of Day', 'Precipitation', 'Relative Humidity', 'Specific Humidity', 'Temperature', 'u Wind', 'v Wind']))


############ Data Exploration for original data  ############


visualize scatter matrix for each cell
plt.style.use('ggplot')
sns.set_style("ticks")
for i in range(len(binary_labels)):
    sns.pairplot(combined_dfs[i], hue="Crash")
    plt.show()


# Logistic regression
params = [10**p for p in range(-7, 7)]

models, performance = Opt.Cross_Validate_Multiple(CM.Logistic, pre_unpaired_data, binary_labels, params)

# Random forest
n_estimators = [10**p for p in range(1,4)]
max_depth = [1 ,2, 4, 6, 10, 15, None]
params = [n_estimators, max_depth]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.RandomForestCl, pre_unpaired_data, binary_labels, params)


# SVM
C = [10**p for p in range(-5,4)]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
params = [C, kernel]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.SVM, pre_unpaired_data, binary_labels, params)
    
#NNC
layers = [2,3]
activations = ['relu']
epochs = [2, 4, 6, 8, 10]
batch_size = [2, 5, 10, 20]
weights = [{0:1, 1:100}]
params = [layers, activations, epochs, batch_size, weights]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.NNC, pre_unpaired_data, binary_labels, params)

########### Data Exploration for upsampled data  ############

# Upsample data (for visualization only)
upsampled_dfs = []
for df in combined_dfs:
    df_nc = df[df['Crash']==0]
    df_c = df[df['Crash']==1]
    df_c_upsampled = resample(df_c, replace=True, n_samples=len(df_nc))   
    df_upsampled = pd.concat([df_nc, df_c_upsampled])
    upsampled_dfs.append(df_upsampled)

visualize scatter matrix for each cell
plt.style.use('ggplot')
sns.set_style("ticks")
for i in range(len(binary_labels)):
    sns.pairplot(upsampled_dfs[i], hue="Crash")
    plt.show()


# Logistic regression
params = [10**p for p in range(-7, 7)]

models, performance = Opt.Cross_Validate_Multiple(CM.Logistic, pre_unpaired_data, binary_labels, params, upsample = 1)


# Random forest
n_estimators = [10**p for p in range(1,4)]
max_depth = [1 ,2, 3, 4, 5, 6, None]
params = [n_estimators, max_depth]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.RandomForestCl, pre_unpaired_data, binary_labels, params, upsample = 1)

#SVM
C = [10**p for p in range(-5,4)]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
params = [C, kernel]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.SVM, pre_unpaired_data, binary_labels, params, upsample = 1)

#NNC
layers = [2,3]
activations = ['relu']
epochs = [2, 4, 6, 8, 10]
batch_size = [2, 5, 10, 20]
weights = [{0:1, 1:100}]
params = [layers, activations, epochs, batch_size, weights]
params = Opt.Param_Permutations(params)

models, performance = Opt.Cross_Validate_Multiple(CM.NNC, pre_unpaired_data, binary_labels, params, upsample = 1)

