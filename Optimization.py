import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

def Cross_Validation_Single(model, X, Y, K = 5):
#	Performs a single iteration of K-Fold cross validation with a single model

	skf = KFold(n_splits=K, shuffle = True)

	performance = 0
	for train_idx, val_idx in skf.split(X):
		model.Train(X[train_idx], Y[train_idx])
		performance += model.Eval(X[val_idx], Y[val_idx])

	performance = performance/K

	return performance

def Cross_Validation_Tune(modelclass, X, Y, params, K = 5):

	# First iteration
	combinations = params[0]
	for i in range(1, len(params)):
		combinations = itertools.product(combinations, params[i])

	combinations = list(combinations)
	performance = np.zeros(len(combinations))
	models = []

	for i,comb in enumerate(combinations):
		model = modelclass(comb)
		models.append(model)
		performance[i] = Cross_Validation_Single(model, X, Y, K)

	best_m = np.argmax(performance)

	return models[best_m], performance[best_m]
