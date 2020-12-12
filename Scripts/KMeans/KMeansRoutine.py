import numpy as np
import pandas as pd
import importlib
import sys
from sklearn.model_selection import KFold

sys.path.insert(1, '.')

import Support as Sup
import Clustering as Cl
import Optimization as Opt
import PreProcessing as Proc

kmeans = Cl.Kmeans(plot = 0)
crashes = Sup.Load_Crash_Data()
crash_locations = np.delete(crashes, 0, 1).astype(float)

performance = 0
for n in range(10):
    skf = KFold(n_splits=5)
    for train_idx, val_idx in skf.split(crash_locations):
        kmeans.Train(crash_locations[train_idx], crashes[train_idx])
        performance += Sup.Competition_Metric([kmeans.centers], crashes[val_idx], time_dep =0)

performance = performance/50


#kmeans.Train(crash_locations, crashes)
#performance = Sup.Competition_Metric([kmeans.centers], crashes, time_dep =0)
print("Sum of distance with K-Mean: " + str(performance))