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


## Generate data structs from the start (merge segment and weather data)
segments_merged = Sup.Load_Merge_Road()
segments_paired, inds_unpaired = Proc.Merge_Roads_Crashes(segments_merged, crash_locations, crashes)
full_paired_data, paired_crash_labels = Proc.Create_Paired_Series(segments_paired, weather_data)


## Generate data structs from the start (Only use road segment data)
simple_paired_data, simple_crash_labels = Proc.Create_Paired_Series2(segments_paired)


## Save data
segments_paired.to_pickle('Scripts/ProcessData/segments_paired.pkl')
np.save('Scripts/ProcessData/inds_unpaired.npy', inds_unpaired)
np.save('Scripts/ProcessData/full_paired_data.npy', full_paired_data)
np.save('Scripts/ProcessData/paired_crash_labels.npy', paired_crash_labels)
np.save('Scripts/ProcessData/simple_paired_data.npy', simple_paired_data)
np.save('Scripts/ProcessData/simple_crash_labels.npy', simple_crash_labels)

