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

from KMeansRoutine import KMeansRoutine
from CreatePairedData import CreatePairedData
from CreateUnpairedData import CreateUnpairedData
from ExploreUnpairedData import ExploreUnpairedData
from TrainPairedData import TrainPairedData

## The full code base is made up of many independent scripts and supporting classes and functions. 
## This main script runs every script in the order which they were run throughout the project

# Run K-means to establish baseline
KMeansRoutine()

# Data preprocessing
CreatePairedData()
CreateUnpairedData()

# Data exploration and training on unpaired dataset
ExploreUnpairedData()

# Feature selection, Training, and model selection on paired dataset
TrainPairedData()


