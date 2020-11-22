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

# Load data
crashes = Sup.Load_Crash_Data()
crash_locations = np.delete(crashes, 0, 1).astype(float)
crash_locations[:, [1, 0]] = crash_locations[:, [0, 1]]
weather = pd.read_csv('Data/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])
predictive_data = np.load('Routines/Routine 1/predictive_data.npy')
crash_labels = np.load('Routines/Routine 1/crash_labels.npy')

# Feature Selection
trimmed = FS.Nan_Threshold(predictive_data, .4)
trimmed_20 = FS.Top_Variance_Select(trimmed, 20)