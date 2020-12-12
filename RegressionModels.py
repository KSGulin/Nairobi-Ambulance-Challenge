from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from PreProcessing import Upsample
import numpy as np
import pandas as pd

class RoundedRidge:
#	Simple K-Means approach for ambulance placement
#	The flag "plot" is used to turn on/off plotting

    def __init__(self, alpha):
        self.alpha = alpha
        self.model = Ridge(alpha = self.alpha)

    def Train(self, X, Y):
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        prediction = np.round(self.model.predict(X))
        return np.sum(prediction == Y)/Y.size

class RoundedLasso:
    def __init__(self, params, upsample = 0):
        self.alpha = params[0]
        self.standardize = params[1]
        self.upsample = upsample
        self.model = Lasso(alpha = self.alpha)

    def Train(self, X, Y):
        if self.standardize:
            X = StandardScaler().fit_transform(X)
        if self.upsample:
            X, Y = Upsample(X, Y)
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        predictions = np.round(self.model.predict(X))
        return -np.mean(np.square(predictions - Y))

class RandomForrestReg:
    def __init__(self, params, upsample = 0):
        self.n_estimators = params[0]
        self.max_depth = params[1]
        self.upsample = upsample
        self.model = RandomForestRegressor(n_estimators = self.n_estimators, max_depth = self.max_depth)

    def Train(self, X, Y):
        if self.upsample:
            X, Y = Upsample(X, Y)
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        predictions = np.round(self.model.predict(X))
        return -np.mean(np.square(predictions - Y))
	
