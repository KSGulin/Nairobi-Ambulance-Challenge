import numpy as np
from sklearn.model_selection import KFold

def Cross_Validation_Single(model, X, Y, K):
#	Performs a single iteration of K-Fold cross validation with a single model

	skf = KFold(n_splits=K)

	performance = 0
	for train_idx, val_idx in skf.split(X):
		model.Train(X[train_idx], Y[train_idx], 1)
		performance += model.Eval(X[val_idx], Y[val_idx])

	performance = performance/K

	return performance