from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
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

class Logistic:
#	Simple K-Means approach for ambulance placement
#	The flag "plot" is used to turn on/off plotting

    def __init__(self, C):
        self.C= C
        self.model = LogisticRegression(C = self.C)

    def Train(self, X, Y):
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        prediction = self.model.predict(X)
        P = np.argwhere(prediction == True)
        if len(P) == 0:
            return 0
        return np.sum(prediction[P] == Y[P])/np.sum(prediction[P])
	
