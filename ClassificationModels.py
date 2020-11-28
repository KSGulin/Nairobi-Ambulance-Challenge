from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

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


class RandomForestCl:
    def __init__(self, params):
        self.n_estimators = params[0]
        self.max_depth = params[1]
        self.model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth)

    def Train(self, X, Y):
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        prediction = self.model.predict(X)
        P = np.argwhere(prediction == True)
        if len(P) == 0:
            return 0
        return np.sum(prediction[P] == Y[P])/np.sum(prediction[P])