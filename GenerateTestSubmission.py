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

predicted_crashes = pd.DataFrame() 
predicted_crashes['Crash_Locations'] = np.empty((8, 0)).tolist()
predicted_crashes['Crash_Weights'] = np.empty((8, 0)).tolist()

crashes = Sup.Load_Crash_Data()
crash_locations = np.delete(crashes, 0, 1).astype(float)
crash_locations[:, [1, 0]] = crash_locations[:, [0, 1]]
inds_unpaired = np.load('Scripts/ProcessData/inds_unpaired.npy')

## Add unpaired data static "predictions"
for ind in inds_unpaired:
    h_ind = int(np.floor(crashes[ind][0].hour/3))
    predicted_crashes.iloc[h_ind,0].append(crash_locations[ind])
    predicted_crashes.iloc[h_ind,1].append(1/546)


## Get average crash location per road segment. Use road midpoint if there is no crash data
segments_paired = pd.read_pickle('ProcessedData/segments_paired.pkl')
avg_crash_loc = np.zeros((1535, 2))

for i, row in segments_paired.iterrows():
    if row[4]:
        avg_crash_loc[i] = np.mean(row[4], axis = 0)
    else:
        centroid = row[2].centroid
        avg_crash_loc[i] = [centroid.x, centroid.y]

## Predict with model trained on paired data

simple_trimmed_data = np.load('ProcessedData/simple_trimmed_data.npy')
simple_crash_labels = np.load('ProcessedData/simple_crash_labels.npy')


params = [100, 50]
model = RM.RandomForrestReg(params)

# Estimate model performance
MSE = Opt.Estimate_Performance_Reg(model, simple_trimmed_data, simple_crash_labels)

## fill in predicted crash data
model.Train(simple_trimmed_data, simple_crash_labels)
predicted_labels = np.round(model.model.predict(simple_trimmed_data)).astype(int)


for i, label in enumerate(predicted_labels):
    ind = int(np.floor(i/8))
    for j in range(label):
        predicted_crashes.iloc[int(simple_trimmed_data[i,0]),0].append(avg_crash_loc[ind])
        predicted_crashes.iloc[int(simple_trimmed_data[i,0]),1].append(1/546)

## Generate ambulance locations. Unweighted version is used because all weights are indetical in this prediction
kmeans = Cl.Kmeans(plot = 0)
ambulance_loc = np.zeros((8, 12))
for i in range(8):
    kmeans.Train(predicted_crashes.iloc[i,0], predicted_crashes.iloc[i,0])
    ambulance_loc[i] = kmeans.centers.flatten()


## print submission file
ss = pd.read_csv('Data/SampleSubmission.csv', parse_dates=['date'])
ss.head()
d = pd.DataFrame([])
d['date'] = ss['date'] 
ambulance_print = np.tile(ambulance_loc, (184,1))
flip_inds = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
for i in range(12):
    d[ss.keys()[i+1]] = ambulance_print[:,flip_inds[i]]
d.head()
d.to_csv('Data/submission.csv')