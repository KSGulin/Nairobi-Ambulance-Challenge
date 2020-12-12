import numpy as np
import pandas as pd
import importlib
import sys

sys.path.insert(1, '.')

import Support as Sup
import PreProcessing as Proc

## Load raw data
crashes = Sup.Load_Crash_Data()
crash_locations = np.delete(crashes, 0, 1).astype(float)
crash_locations[:, [1, 0]] = crash_locations[:, [0, 1]]
weather = pd.read_csv('Data/Weather_Nairobi_Daily_GFS_interpolated.csv', parse_dates=['Date'])
weather_data = weather.iloc[:546,]
weather_test = weather.iloc[546:,]

## Load preprocessed data
inds_unpaired = np.load('Scripts/ProcessData/inds_unpaired.npy')

## Create samples of unpaired data
grid = Proc.Create_Grid(crashes[inds_unpaired,:], .1)
flat_grid = [j for sub in grid for j in sub]
full_unpaired_data, unpaired_crash_labels, active_cells = Proc.Create_Unpaired_Series(flat_grid, weather_data)


## Break off 3 regions for data analysis and trying out different techniques
pre_idx = np.zeros(3)
for i in range(3):
    pre_idx[i] = int(np.random.randint(0, unpaired_crash_labels.shape[0]))

pre_idx = np.asarray([5, 27, 53])
pre_unpaired_data = full_unpaired_data[pre_idx.astype(int)]
pre_unpaired_labels = unpaired_crash_labels[pre_idx]

train_unpaired_data = np.delete(full_unpaired_data, pre_idx, axis = 0)
train_unpaired_labels = np.delete(unpaired_crash_labels, pre_idx, axis = 0)

## Save data
np.save('Scripts/ProcessData/full_unpaired_data.npy', full_unpaired_data)
np.save('Scripts/ProcessData/unpaired_crash_labels.npy', unpaired_crash_labels)
np.save('Scripts/ProcessData/pre_unpaired_data.npy', pre_unpaired_data)
np.save('Scripts/ProcessData/pre_unpaired_labels.npy', pre_unpaired_labels)
np.save('Scripts/ProcessData/train_unpaired_data.npy', train_unpaired_data)
np.save('Scripts/ProcessData/train_unpaired_labels.npy', train_unpaired_labels)