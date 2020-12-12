import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from keras.metrics import Precision
from keras.models import Sequential
from keras.layers import Dense
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler


from Support import Precision_Calc
from PreProcessing import Upsample

class Logistic:

    def __init__(self, C, upsample = 0):
        self.C= C
        self.upsample = upsample
        self.model = LogisticRegression(C = self.C)

    def Train(self, X, Y):
        if self.upsample:
            X, Y = Upsample(X, Y)
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        predictions = self.model.predict(X)
        try:
            return roc_auc_score(Y, predictions)
        except ValueError:
            return .5



class RandomForestCl:
    def __init__(self, params, upsample = 0):
        self.n_estimators = params[0]
        self.max_depth = params[1]
        self.upsample = upsample
        self.model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth)

    def Train(self, X, Y):
        if self.upsample:
            X, Y = Upsample(X, Y, .1)
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        predictions = self.model.predict(X)
        try:
            return roc_auc_score(Y, predictions)
        except ValueError:
            return .5


class SVM:
    def __init__(self, params, upsample = 0):
        self.C= params[0]
        self.kernel = params[1]
        self.upsample = upsample
        self.model = SVC(C = self.C, kernel = self.kernel, class_weight = 'balanced')

    def Train(self, X, Y):
        if self.upsample:
            X, Y = Upsample(X, Y)
        self.model.fit(X, Y)

    def Eval(self, X, Y):
        predictions = self.model.predict(X)
        try:
            return roc_auc_score(Y, predictions)
        except ValueError:
            return .5


class NNC:
    def __init__(self, params, upsample = 0):
        self.layers = params[0]
        self.activation = params[1]
        self.epochs = params[2]
        self.batch_size = params[3]
        self.class_weights = params[4]
        self.upsample = upsample
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=7, activation=self.activation))
        if (self.layers > 2):
            self.model.add(Dense(4, activation=self.activation))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision()])

    def Train(self, X, Y):
        if self.upsample:
            X, Y = Upsample(X, Y, .1)
        self.model.fit(X, Y, class_weight = self.class_weights, epochs = self.epochs, batch_size = self.batch_size)

    def Eval(self, X, Y):
        predictions = self.model.predict(X)
        try:
            return roc_auc_score(Y, predictions)
        except ValueError:
            return .5